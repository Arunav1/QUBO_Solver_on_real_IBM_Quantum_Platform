
from itertools import product
from typing import Any


def calculate_objective(model, assignment: dict[str, int]) -> float:
    """Calculate the original objective function."""
    return float(
        sum(
            float(coefficient) * assignment[variable]
            for variable, coefficient in model.objective.items()
        )
    )



def check_constraints(model, assignment: dict[str, int]) -> dict:
    """Check every supported constraint against the original model."""
    results = {}
    tol = 1e-9

    for constraint in model.constraints:
        lhs = sum(
            float(coefficient) * assignment[variable]
            for variable, coefficient in constraint.coefficients.items()
        )
        rhs = float(constraint.rhs)
        sense = constraint.sense

        if sense == "==":
            satisfied = abs(lhs - rhs) <= tol
        elif sense == "<=":
            satisfied = lhs <= rhs + tol
        elif sense == ">=":
            satisfied = lhs >= rhs - tol
        elif sense == "<":
            satisfied = lhs < rhs - tol
        elif sense == ">":
            satisfied = lhs > rhs + tol
        else:
            raise ValueError(
                f"Unsupported constraint sense: {sense}"
            )

        results[constraint.name] = {
            "lhs": lhs,
            "sense": sense,
            "rhs": rhs,
            "satisfied": satisfied,
        }

    return {
        "feasible": all(
            item["satisfied"] for item in results.values()
        ),
        "constraints": results,
    }



def calculate_qubo_energy(
    qubo: dict[str, Any],
    assignment: dict[str, int],
) -> float:
    """Evaluate the QUBO energy for a binary assignment."""
    energy = float(qubo.get("constant", 0.0))

    for term in qubo["terms"]:
        i = term["var_i"]
        j = term["var_j"]
        coefficient = float(term["coefficient"])

        energy += coefficient * assignment[i] * assignment[j]

    return energy


def exact_solve_original(model) -> dict:
    """
    Enumerate all binary assignments and find the best feasible
    solution to the ORIGINAL optimization problem.

    Intended for small problems only.
    """
    variables = list(model.variables)

    if len(variables) > 20:
        raise ValueError(
            "Exact verification supports at most 20 variables."
        )

    best_assignment = None
    best_objective = None
    feasible_count = 0

    for bits in product([0, 1], repeat=len(variables)):
        assignment = dict(zip(variables, bits))

        constraint_result = check_constraints(model, assignment)

        if not constraint_result["feasible"]:
            continue

        feasible_count += 1
        objective = calculate_objective(model, assignment)

        if best_objective is None:
            better = True
        elif model.sense == "min":
            better = objective < best_objective
        else:
            better = objective > best_objective

        if better:
            best_objective = objective
            best_assignment = assignment

    if best_assignment is None:
        return {
            "feasible_solution_exists": False,
            "message": "No feasible binary assignment exists.",
        }

    return {
        "feasible_solution_exists": True,
        "optimal_assignment": best_assignment,
        "optimal_objective": best_objective,
        "feasible_assignments_checked": feasible_count,
    }


def verify_solution(
    model,
    qubo: dict[str, Any],
    assignment: dict[str, int],
) -> dict:
    """Verify a candidate against the original model and, when possible, the QUBO."""

    variables = list(model.variables)

    missing = set(variables) - set(assignment)
    if missing:
        return {
            "verified": False,
            "error": "Assignment is missing original model variables.",
            "missing_variables": sorted(missing),
        }

    original_assignment = {
        variable: assignment[variable]
        for variable in variables
    }

    if any(value not in (0, 1) for value in assignment.values()):
        return {
            "verified": False,
            "error": "Every assignment value must be 0 or 1.",
        }

    qubo_variables = set(qubo.get("variables", []))
    model_variables = set(variables)

    if not model_variables.issubset(qubo_variables):
        return {
            "verified": False,
            "error": "QUBO is missing original model variables.",
        }

    constraint_result = check_constraints(model, original_assignment)
    objective = calculate_objective(model, original_assignment)

    # QUBO energy requires every original and auxiliary variable.
    complete_qubo_assignment = (
        set(assignment) == qubo_variables
    )

    qubo_energy = None
    if complete_qubo_assignment:
        qubo_energy = calculate_qubo_energy(qubo, assignment)

    try:
        exact_result = exact_solve_original(model)
    except ValueError as error:
        exact_result = {
            "feasible_solution_exists": None,
            "available": False,
            "reason": str(error),
        }

    optimal = None

    if exact_result.get("feasible_solution_exists") is True:
        optimum = exact_result["optimal_objective"]

        if model.sense == "min":
            optimal = objective <= optimum + 1e-9
        else:
            optimal = objective >= optimum - 1e-9

    feasible = constraint_result["feasible"]

    return {
        "verified": feasible and optimal is True,
        "feasible": feasible,
        "optimal": optimal,
        "assignment": original_assignment,
        "original_objective": objective,
        "qubo_energy": qubo_energy,
        "qubo_energy_available": complete_qubo_assignment,
        "constraint_check": constraint_result,
        "exact_reference": exact_result,
    }
