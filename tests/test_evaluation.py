import copy
import json
import time
from pathlib import Path

from backend.schemas import OptimizationModel
from backend.qubo_compiler import compile_qubo, exact_solve
from backend.qubo_verifier import verify_qubo_equivalence
from backend.repair_loop import repair_qubo
from backend.qaoa_solver import solve_qaoa
from backend.verifier import verify_solution


OUTPUT_DIR = Path("evaluation_results")
OUTPUT_DIR.mkdir(exist_ok=True)


# --------------------------------------------------
# 1. Benchmark optimization problems
# --------------------------------------------------

BENCHMARKS = [
    {
        "name": "select_one",
        "model": {
            "title": "Select the cheaper item",
            "variables": ["x_A", "x_B"],
            "objective": {"x_A": 3, "x_B": 2},
            "sense": "min",
            "constraints": [
                {
                    "name": "select_exactly_one",
                    "coefficients": {"x_A": 1, "x_B": 1},
                    "sense": "==",
                    "rhs": 1,
                }
            ],
        },
    },
    {
        "name": "maximize_value",
        "model": {
            "title": "Choose one item with maximum value",
            "variables": ["x_A", "x_B", "x_C"],
            "objective": {"x_A": 5, "x_B": 8, "x_C": 4},
            "sense": "max",
            "constraints": [
                {
                    "name": "choose_one",
                    "coefficients": {"x_A": 1, "x_B": 1, "x_C": 1},
                    "sense": "==",
                    "rhs": 1,
                }
            ],
        },
    },
    {
        "name": "two_selected",
        "model": {
            "title": "Select two items with minimum cost",
            "variables": ["x_A", "x_B", "x_C"],
            "objective": {"x_A": 4, "x_B": 1, "x_C": 3},
            "sense": "min",
            "constraints": [
                {
                    "name": "select_two",
                    "coefficients": {"x_A": 1, "x_B": 1, "x_C": 1},
                    "sense": "==",
                    "rhs": 2,
                }
            ],
        },
    },
]


def make_model(data):
    """Support Pydantic v2 and v1."""
    if hasattr(OptimizationModel, "model_validate"):
        return OptimizationModel.model_validate(data)
    return OptimizationModel.parse_obj(data)


def model_to_dict(model):
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


# --------------------------------------------------
# 2. Test QUBO correctness and exact solution
# --------------------------------------------------

def evaluate_benchmark(benchmark):
    started = time.perf_counter()
    result = {
        "name": benchmark["name"],
        "passed": False,
        "checks": {},
    }

    try:
        model = make_model(benchmark["model"])
        qubo = compile_qubo(model, penalty=10.0)

        equivalence = verify_qubo_equivalence(
            model,
            qubo,
            penalty=10.0,
            max_variables=20,
            tolerance=1e-8,
        )

        exact = exact_solve(model, qubo)

        result["checks"]["qubo_equivalent"] = bool(
            equivalence["equivalent"]
        )
        result["checks"]["exact_solution_exists"] = exact is not None
        result["equivalence"] = equivalence
        result["qubo"] = qubo

        if exact is not None:
            result["exact_solution"] = exact

        result["passed"] = (
            result["checks"]["qubo_equivalent"]
            and result["checks"]["exact_solution_exists"]
        )

    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"

    result["runtime_seconds"] = round(
        time.perf_counter() - started, 4
    )
    return result


# --------------------------------------------------
# 3. Deliberately corrupt a QUBO
# --------------------------------------------------

def evaluate_corruption(benchmark):
    model = make_model(benchmark["model"])
    penalty = 10.0

    correct_qubo = compile_qubo(model, penalty=penalty)
    faulty_qubo = copy.deepcopy(correct_qubo)

    # Change the first coefficient to simulate a compiler error.
    if not faulty_qubo["terms"]:
        return {
            "name": benchmark["name"],
            "passed": False,
            "error": "No QUBO terms available to corrupt.",
        }

    faulty_qubo["terms"][0]["coefficient"] += 7.0

    before = verify_qubo_equivalence(
        model,
        faulty_qubo,
        penalty=penalty,
        max_variables=20,
        tolerance=1e-8,
    )

    repair_result = repair_qubo(
        model,
        faulty_qubo,
        penalty=penalty,
        max_variables=20,
        tolerance=1e-8,
    )

    # Accommodate common repair result formats.
    repaired_qubo = repair_result.get("repaired_qubo")

    if repaired_qubo is None:
        repaired_qubo = repair_result.get("qubo")

    after = repair_result.get("verification")

    if after is None:
        after = repair_result.get("final_verification")

    # If the repair function doesn't include its verification,
    # independently verify its returned QUBO.
    if after is None and repaired_qubo is not None:
        after = verify_qubo_equivalence(
            model,
            repaired_qubo,
            penalty=penalty,
            max_variables=20,
            tolerance=1e-8,
        )

    detected = not before["equivalent"]
    repaired = (
        repaired_qubo is not None
        and after is not None
        and after.get("equivalent", False)
    )

    return {
        "name": benchmark["name"],
        "corruption_detected": detected,
        "repair_succeeded": bool(repaired),
        "passed": detected and bool(repaired),
        "before": before,
        "repair": repair_result,
        "after": after,
    }


# --------------------------------------------------
# 4. Test QAOA and independent solution verification
# --------------------------------------------------

def evaluate_qaoa(benchmark):
    started = time.perf_counter()
    model = make_model(benchmark["model"])
    penalty = 10.0

    qubo = compile_qubo(model, penalty=penalty)

    equivalence = verify_qubo_equivalence(
        model,
        qubo,
        penalty=penalty,
        max_variables=20,
        tolerance=1e-8,
    )

    # Never run QAOA on an unverified QUBO.
    if not equivalence["equivalent"]:
        return {
            "name": benchmark["name"],
            "passed": False,
            "error": "QUBO equivalence verification failed.",
        }

    qaoa = solve_qaoa(
        qubo,
        p=1,
        shots=2048,
        maxiter=100,
        seed=42,
    )

    best = qaoa.get("best_sample", {})
    assignment = best.get("assignment")

    if assignment is None:
        return {
            "name": benchmark["name"],
            "passed": False,
            "error": "QAOA returned no best assignment.",
            "qaoa": qaoa,
        }

    verification = verify_solution(
        model,
        qubo,
        assignment,
    )

    result = {
        "name": benchmark["name"],
        "qubo_equivalent": True,
        "qaoa": qaoa,
        "solution_verification": verification,
        "passed": bool(
            verification.get("verified", False)
            and verification.get("feasible", False)
            and verification.get("optimal", False)
        ),
        "runtime_seconds": round(
            time.perf_counter() - started, 4
        ),
    }

    return result


# --------------------------------------------------
# 5. Run the complete evaluation
# --------------------------------------------------

def main():
    report = {
        "project": "VeriQAOA",
        "benchmark_count": len(BENCHMARKS),
        "benchmark_results": [],
        "corruption_results": [],
        "qaoa_results": [],
    }

    for benchmark in BENCHMARKS:
        print(f"\nEvaluating: {benchmark['name']}")

        benchmark_result = evaluate_benchmark(benchmark)
        report["benchmark_results"].append(benchmark_result)

        print(
            "QUBO and exact checks:",
            "PASS" if benchmark_result["passed"] else "FAIL",
        )

        try:
            corruption_result = evaluate_corruption(benchmark)
            report["corruption_results"].append(corruption_result)

            print(
                "Corruption and repair:",
                "PASS" if corruption_result["passed"] else "FAIL",
            )
        except Exception as error:
            report["corruption_results"].append({
                "name": benchmark["name"],
                "passed": False,
                "error": f"{type(error).__name__}: {error}",
            })
            print("Corruption and repair: ERROR")

        try:
            qaoa_result = evaluate_qaoa(benchmark)
            report["qaoa_results"].append(qaoa_result)

            print(
                "QAOA and solution verification:",
                "PASS" if qaoa_result.get("passed") else "FAIL",
            )
        except Exception as error:
            report["qaoa_results"].append({
                "name": benchmark["name"],
                "passed": False,
                "error": f"{type(error).__name__}: {error}",
            })
            print("QAOA evaluation: ERROR")

    # Summarize outcomes.
    for key in (
        "benchmark_results",
        "corruption_results",
        "qaoa_results",
    ):
        results = report[key]
        report[key + "_summary"] = {
            "passed": sum(bool(item.get("passed")) for item in results),
            "total": len(results),
        }

    output_file = OUTPUT_DIR / "evaluation_report.json"
    output_file.write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )

    print("\nEvaluation complete.")
    print(json.dumps({
        key: report[key]
        for key in (
            "benchmark_results_summary",
            "corruption_results_summary",
            "qaoa_results_summary",
        )
    }, indent=2))
    print(f"\nReport saved to: {output_file.resolve()}")


if __name__ == "__main__":
    main()