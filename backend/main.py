
import os
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from openai import OpenAIError

from dotenv import load_dotenv

load_dotenv()

import logging

logger = logging.getLogger(__name__)

from .schemas import OptimizationModel
from .constraints import suggest_penalty
from .qubo_compiler import compile_qubo, exact_solve
from .verifier import verify_solution
from .qaoa_solver import optimize_qaoa, solve_qaoa, summarize_counts
from .ibm_runtime import (
    fetch_job_counts,
    ibm_configured,
    submit_sampler_job,
)
from .job_store import load_job, save_job, update_job
from .qubo_verifier import verify_qubo_equivalence
from .repair_loop import repair_qubo
from .llm_formulator import formulate_optimization_problem

app = FastAPI(title="VeriQAOA API")


def resolve_penalty(model: OptimizationModel, penalty: float | None) -> float:
    """Use the caller's penalty, or one large enough to keep every
    infeasible assignment above the feasible optimum."""
    return penalty if penalty is not None else suggest_penalty(model)


Execution = Literal["ibm", "simulator"]


def resolve_execution(execution: Execution | None) -> Execution:
    """Request choice, then QUANTUM_EXECUTION, then IBM if a key is set."""
    if execution:
        return execution

    configured = os.getenv("QUANTUM_EXECUTION", "").strip().lower()
    if configured in ("ibm", "simulator"):
        return configured

    return "ibm" if ibm_configured() else "simulator"


class CompileRequest(BaseModel):
    model: OptimizationModel
    penalty: float | None = Field(default=None, gt=0)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/compile")
def compile_model(request: CompileRequest):
    try:
        qubo = compile_qubo(
            request.model,
            resolve_penalty(request.model, request.penalty)
        )

        solution = exact_solve(
            request.model,
            qubo
        )

        return {
            "qubo": qubo,
            "solution": solution
        }

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

class VerifyRequest(BaseModel):
    model: OptimizationModel
    qubo: dict
    assignment: dict[str, int]


@app.post("/api/verify")
def verify_model(request: VerifyRequest):
    try:
        return verify_solution(
            model=request.model,
            qubo=request.qubo,
            assignment=request.assignment,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise HTTPException(
            status_code=400,
            detail=f"Verification failed: {error}",
        ) from error
        

class SolveRequest(BaseModel):
    model: OptimizationModel
    qubo: dict
    p: int = Field(default=1, ge=1, le=5)
    shots: int = Field(default=2048, ge=1, le=100000)
    maxiter: int = Field(default=100, ge=1, le=1000)


@app.post("/api/solve")
def solve_model(request: SolveRequest):
    try:
        qaoa_result = solve_qaoa(
            qubo=request.qubo,
            p=request.p,
            shots=request.shots,
            maxiter=request.maxiter,
            decision_variables=request.model.variables,
        )

        candidate = qaoa_result["best_sample"]["assignment"]

        verification = verify_solution(
            model=request.model,
            qubo=request.qubo,
            assignment=candidate,
        )

        return {
            "qaoa": qaoa_result,
            "verification": verification,
        }

    except (ValueError, KeyError, TypeError) as error:
        raise HTTPException(
            status_code=400,
            detail=f"QAOA solve failed: {error}",
        ) from error


class QUBOVerifyRequest(BaseModel):
    model: OptimizationModel
    qubo: dict
    penalty: float | None = Field(default=None, gt=0)
    max_variables: int = Field(default=20, ge=1, le=20)
    tolerance: float = Field(default=1e-8, ge=0)


@app.post("/api/verify-qubo")
def verify_qubo(request: QUBOVerifyRequest):
    try:
        return verify_qubo_equivalence(
            model=request.model,
            qubo=request.qubo,
            penalty=resolve_penalty(request.model, request.penalty),
            max_variables=request.max_variables,
            tolerance=request.tolerance,
        )
    except (ValueError, KeyError, TypeError) as error:
        raise HTTPException(
            status_code=400,
            detail=f"QUBO verification failed: {error}",
        ) from error


class RepairRequest(BaseModel):
    model: OptimizationModel
    qubo: dict
    penalty: float | None = Field(default=None, gt=0)
    max_variables: int = Field(default=20, ge=1, le=20)
    tolerance: float = Field(default=1e-8, ge=0)


@app.post("/api/repair-qubo")
def repair_qubo_endpoint(request: RepairRequest):
    try:
        return repair_qubo(
            model=request.model,
            faulty_qubo=request.qubo,
            penalty=resolve_penalty(request.model, request.penalty),
            max_variables=request.max_variables,
            tolerance=request.tolerance,
        )
    except (ValueError, KeyError, TypeError) as error:
        raise HTTPException(
            status_code=400,
            detail=f"QUBO repair failed: {error}",
        ) from error


class VerifiedOptimizeRequest(BaseModel):
    model: OptimizationModel
    penalty: float | None = Field(default=None, gt=0)
    p: int = Field(default=1, ge=1, le=5)
    shots: int = Field(default=2048, ge=1, le=100000)
    maxiter: int = Field(default=100, ge=1, le=1000)
    max_variables: int = Field(default=20, ge=1, le=20)
    tolerance: float = Field(default=1e-8, ge=0)
    execution: Execution | None = None


@app.post("/api/optimize-verified")
def optimize_verified(request: VerifiedOptimizeRequest):
    try:
        penalty = resolve_penalty(request.model, request.penalty)

        # Stage 1: Compile the original model.
        qubo = compile_qubo(
            request.model,
            penalty,
        )

        qubit_count = len(qubo["variables"])
        if qubit_count > request.max_variables:
            slack_count = qubit_count - len(request.model.variables)
            raise ValueError(
                f"The compiled QUBO needs {qubit_count} qubits "
                f"({len(request.model.variables)} decision variables + "
                f"{slack_count} slack variables for inequality "
                f"constraints), but at most {request.max_variables} "
                "can be simulated. Reduce the number of variables or "
                "the size of the inequality bounds."
            )

        # Stage 2: Independently verify the compiled QUBO.
        initial_verification = verify_qubo_equivalence(
            model=request.model,
            qubo=qubo,
            penalty=penalty,
            max_variables=request.max_variables,
            tolerance=request.tolerance,
        )

        repair_result = None

        # Stage 3: Repair and verify again if necessary.
        if not initial_verification["equivalent"]:
            repair_result = repair_qubo(
                model=request.model,
                faulty_qubo=qubo,
                penalty=penalty,
                max_variables=request.max_variables,
                tolerance=request.tolerance,
            )

            if repair_result["repair_status"] not in (
                "repaired",
                "not_needed",
            ):
                return {
                    "status": "verification_failed",
                    "message": (
                        "QUBO verification failed after repair. "
                        "QAOA was not executed."
                    ),
                    "initial_verification": initial_verification,
                    "repair": repair_result,
                }

            qubo = repair_result["repaired_qubo"]

        # Always verify the exact QUBO that will be sent to QAOA.
        final_verification = verify_qubo_equivalence(
            model=request.model,
            qubo=qubo,
            penalty=penalty,
            max_variables=request.max_variables,
            tolerance=request.tolerance,
        )

        if not final_verification["equivalent"]:
            return {
                "status": "verification_failed",
                "message": (
                    "Final QUBO verification failed. "
                    "QAOA was not executed."
                ),
                "initial_verification": initial_verification,
                "final_verification": final_verification,
                "repair": repair_result,
            }

        verified_stages = {
            "penalty": penalty,
            "qubo": qubo,
            "initial_verification": initial_verification,
            "repair": repair_result,
            "final_verification": final_verification,
        }

        # Stage 4 on hardware: optimize the angles classically, then
        # submit the optimized circuit to IBM Quantum. Stage 5 runs
        # when the job finishes (see GET /api/jobs/{job_id}).
        if resolve_execution(request.execution) == "ibm":
            return submit_to_ibm(request, verified_stages)

        # Stage 4: Run QAOA only after equivalence verification passes.
        qaoa_result = solve_qaoa(
            qubo=qubo,
            p=request.p,
            shots=request.shots,
            maxiter=request.maxiter,
            decision_variables=request.model.variables,
        )

        # Stage 5: Verify the candidate against the original model.
        candidate = qaoa_result["best_sample"]["assignment"]

        solution_verification = verify_solution(
            model=request.model,
            qubo=qubo,
            assignment=candidate,
        )

        return {
            "status": "completed",
            **verified_stages,
            "qaoa": qaoa_result,
            "solution_verification": solution_verification,
        }

    except (ValueError, KeyError, TypeError) as error:
        raise HTTPException(
            status_code=400,
            detail=f"Verified optimization failed: {error}",
        ) from error


def submit_to_ibm(request: VerifiedOptimizeRequest, verified_stages: dict):
    prepared = optimize_qaoa(
        qubo=verified_stages["qubo"],
        p=request.p,
        maxiter=request.maxiter,
    )

    try:
        submission = submit_sampler_job(prepared["circuit"], request.shots)
    except ValueError:
        raise
    except Exception as error:
        logger.exception("IBM Quantum submission failed")
        raise HTTPException(
            status_code=502,
            detail=f"IBM Quantum submission failed: {error}",
        ) from error

    save_job(submission["job_id"], {
        "model": request.model.model_dump(),
        "stages": verified_stages,
        "qaoa_metadata": prepared["metadata"],
        "submission": submission,
        "formulation": None,
        "final": None,
    })

    return {
        "status": "queued",
        **verified_stages,
        "ibm_job": {**submission, "state": "QUEUED"},
    }


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    """Poll an IBM Quantum job; finish the pipeline once it is done."""
    try:
        record = load_job(job_id)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    if record is None:
        raise HTTPException(status_code=404, detail="Unknown job id.")

    if record.get("final"):
        return record["final"]

    try:
        job = fetch_job_counts(job_id)
    except Exception as error:
        logger.exception("IBM Quantum job lookup failed")
        raise HTTPException(
            status_code=502,
            detail=f"Could not reach IBM Quantum: {error}",
        ) from error

    submission = record["submission"]
    stages = record["stages"]
    ibm_job = {**submission, "state": job["state"]}

    if "counts" not in job:
        failed = "error" in job
        response = {
            "status": "failed" if failed else job["state"].lower(),
            "formulation": record.get("formulation"),
            "optimization": {
                "status": "failed" if failed else "queued",
                **stages,
                "ibm_job": ibm_job,
            },
        }
        if failed:
            response["message"] = job["error"]
            update_job(job_id, final=response)
        return response

    # Stage 5: Verify the hardware candidate against the original model.
    model = OptimizationModel.model_validate(record["model"])
    qubo = stages["qubo"]

    qaoa_result = summarize_counts(
        qubo,
        job["counts"],
        record["qaoa_metadata"],
        backend=f"IBM Quantum {submission['backend']}",
        decision_variables=model.variables,
    )
    qaoa_result["ibm_job"] = {
        **ibm_job,
        "quantum_seconds": job.get("quantum_seconds"),
    }

    solution_verification = verify_solution(
        model=model,
        qubo=qubo,
        assignment=qaoa_result["best_sample"]["assignment"],
    )

    final = {
        "status": "completed",
        "formulation": record.get("formulation"),
        "optimization": {
            "status": "completed",
            **stages,
            "qaoa": qaoa_result,
            "solution_verification": solution_verification,
        },
    }
    update_job(job_id, final=final)
    return final


class FormulateRequest(BaseModel):
    description: str = Field(min_length=10, max_length=10000)


@app.post("/api/formulate")
def formulate_model(request: FormulateRequest):
    try:
        return formulate_optimization_problem(
            request.description
        )

    except ValueError as error:
        raise HTTPException(
            status_code=422,
            detail=str(error),
        ) from error
    except OpenAIError as error:
        logger.exception("LLM provider request failed")

        raise HTTPException(
            status_code=502,
            detail=(
                f"LLM provider error: "
                f"{type(error).__name__}: {str(error)}"
            ),
        ) from error

class NaturalLanguageOptimizeRequest(BaseModel):
    description: str = Field(min_length=10, max_length=10000)
    penalty: float | None = Field(default=None, gt=0)
    p: int = Field(default=1, ge=1, le=5)
    shots: int = Field(default=2048, ge=1, le=100000)
    maxiter: int = Field(default=100, ge=1, le=1000)
    max_variables: int = Field(default=20, ge=1, le=20)
    tolerance: float = Field(default=1e-8, gt=0)
    execution: Execution | None = None


@app.post("/api/optimize-from-text")
def optimize_from_text(request: NaturalLanguageOptimizeRequest):
    # Step 1: Convert natural language into a structured model.
    try:
        formulation = formulate_optimization_problem(
            request.description
        )
    except ValueError as error:
        raise HTTPException(
            status_code=422,
            detail=str(error),
        ) from error
    except OpenAIError as error:
        # The OpenAI SDK is also used for the Groq-compatible API.
        logger.exception("LLM formulation request failed")
        raise HTTPException(
            status_code=502,
            detail="The LLM provider failed to formulate the problem.",
        ) from error

    # Step 2: Stop if the LLM needs clarification.
    if (
        not formulation.get("formulated")
        or formulation.get("requires_clarification")
    ):
        return {
            "status": "clarification_required",
            "formulation": formulation,
        }

    # Step 3: Validate the generated model.
    try:
        model_data = formulation.get("model")

        if not isinstance(model_data, dict):
            raise ValueError("The LLM did not return a valid model.")

        model = OptimizationModel.model_validate(model_data)

        # Enforce the variable checks used by your compiler.
        model.validate_variables()

    except ValueError as error:
        raise HTTPException(
            status_code=422,
            detail=f"Generated model validation failed: {error}",
        ) from error
    except Exception as error:
        # Handles Pydantic validation errors as well.
        raise HTTPException(
            status_code=422,
            detail=f"Generated model is invalid: {error}",
        ) from error

    # Step 4: Reuse the existing verified optimization pipeline.
    try:
        optimization_request = VerifiedOptimizeRequest(
            model=model,
            penalty=request.penalty,
            p=request.p,
            shots=request.shots,
            maxiter=request.maxiter,
            max_variables=request.max_variables,
            tolerance=request.tolerance,
            execution=request.execution,
        )

        result = optimize_verified(optimization_request)

    except HTTPException as error:
        if error.status_code == 400:
            # Compilation limits (e.g. too many qubits) are a problem
            # with the generated model, not a server failure.
            raise HTTPException(
                status_code=422,
                detail=error.detail,
            ) from error
        raise
    except Exception as error:
        logger.exception("Natural-language optimization failed")
        raise HTTPException(
            status_code=500,
            detail="Optimization failed. Check the server logs.",
        ) from error

    # Step 5: Return the formulation and verified optimization result.
    summary = {
        "title": model.title,
        "assumptions": formulation.get("assumptions", []),
        "model": model.model_dump(),
    }

    # Hardware runs finish later; keep the formulation with the job
    # so polling returns the same report shape.
    if result.get("ibm_job"):
        update_job(result["ibm_job"]["job_id"], formulation=summary)

    return {
        "status": result.get("status", "completed"),
        "formulation": summary,
        "optimization": result,
    }