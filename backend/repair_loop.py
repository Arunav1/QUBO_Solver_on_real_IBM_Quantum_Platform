
from typing import Any

from .constraints import normalize_constraint
from .qubo_verifier import verify_qubo_equivalence


def _slack_weights(max_slack: int) -> list[int]:
    """Binary weights whose subset sums are exactly 0..max_slack."""
    weights = []
    remaining = max_slack
    power = 1

    while remaining > 0:
        weight = min(power, remaining)
        weights.append(weight)
        remaining -= weight
        power *= 2

    return weights


def reconstruct_qubo(model, penalty: float) -> dict[str, Any]:
    """
    Reconstruct the intended QUBO directly from the trusted model.

    Supports binary variables, linear objectives, equality constraints
    and inequality constraints (via binary slack variables), all using
    squared penalties on the normalized integer constraint form.
    """
    if penalty <= 0:
        raise ValueError("Penalty must be positive.")

    variables = list(model.variables)

    if not variables:
        raise ValueError("The model must contain at least one variable.")

    if len(set(variables)) != len(variables):
        raise ValueError("Variable names must be unique.")

    if model.sense not in ("min", "max"):
        raise ValueError(f"Unsupported objective sense: {model.sense}")

    # Store the polynomial as:
    # constant + sum(linear[i] * xi)
    #           + sum(quadratic[i, j] * xi * xj)
    #
    # For binary variables, xi * xi = xi.
    linear = {v: 0.0 for v in variables}
    quadratic = {}
    constant = 0.0

    objective_sign = 1.0 if model.sense == "min" else -1.0

    for variable, coefficient in model.objective.items():
        if variable not in linear:
            raise ValueError(
                f"Objective references unknown variable: {variable}"
            )
        linear[variable] += objective_sign * float(coefficient)

    for index, constraint in enumerate(model.constraints):
        for variable in constraint.coefficients:
            if variable not in model.variables:
                raise ValueError(
                    f"Constraint '{constraint.name}' references "
                    f"unknown variable: {variable}"
                )

        form = normalize_constraint(constraint)
        coefficients = {v: float(a) for v, a in form.coefficients.items()}
        rhs = float(form.rhs)

        if form.sense == "<=":
            # Never violated: no penalty needed.
            if form.max_lhs <= form.rhs:
                continue

            # sum(ai * xi) + slack = rhs, slack in [0, rhs - min_lhs].
            for k, weight in enumerate(
                _slack_weights(form.rhs - form.min_lhs)
            ):
                name = f"__slack_{index}_{constraint.name}_{k}"
                if name in linear:
                    raise ValueError(f"Reserved variable name: {name}")

                variables.append(name)
                linear[name] = 0.0
                coefficients[name] = float(weight)

        # Add P * (sum(ai * xi) - rhs)^2.
        # Constant contribution: P * rhs^2
        constant += penalty * rhs * rhs

        # Linear contribution: P * (ai^2 - 2 * rhs * ai) * xi
        for variable, a in coefficients.items():
            linear[variable] += penalty * (a * a - 2 * rhs * a)

        # Quadratic contribution: 2 * P * ai * aj * xi * xj
        names = [v for v in variables if v in coefficients]

        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                key = (names[i], names[j])

                quadratic[key] = (
                    quadratic.get(key, 0.0)
                    + 2.0 * penalty
                    * coefficients[names[i]] * coefficients[names[j]]
                )

    terms = []

    # Diagonal terms encode linear binary-variable terms.
    for variable in variables:
        coefficient = linear[variable]

        if abs(coefficient) > 1e-12:
            terms.append({
                "var_i": variable,
                "var_j": variable,
                "coefficient": coefficient,
            })

    # Add pairwise terms.
    for (vi, vj), coefficient in quadratic.items():
        if abs(coefficient) > 1e-12:
            terms.append({
                "var_i": vi,
                "var_j": vj,
                "coefficient": coefficient,
            })

    return {
        "variables": variables,
        "constant": constant,
        "terms": terms,
    }


def repair_qubo(
    model,
    faulty_qubo: dict[str, Any],
    penalty: float = 10.0,
    max_variables: int = 20,
    tolerance: float = 1e-8,
) -> dict[str, Any]:
    """
    Verify the supplied QUBO. If it fails, reconstruct it from the
    trusted model and verify the reconstructed result.
    """
    initial_check = verify_qubo_equivalence(
        model=model,
        qubo=faulty_qubo,
        penalty=penalty,
        max_variables=max_variables,
        tolerance=tolerance,
    )

    if initial_check["equivalent"]:
        return {
            "repair_status": "not_needed",
            "original_verification": initial_check,
            "repaired_qubo": faulty_qubo,
            "final_verification": initial_check,
        }

    # A mismatch is our counterexample: it demonstrates that at least
    # one assignment has an incorrect compiled energy.
    counterexamples = initial_check.get("mismatches", [])

    repaired_qubo = reconstruct_qubo(
        model=model,
        penalty=penalty,
    )

    final_check = verify_qubo_equivalence(
        model=model,
        qubo=repaired_qubo,
        penalty=penalty,
        max_variables=max_variables,
        tolerance=tolerance,
    )

    return {
        "repair_status": (
            "repaired"
            if final_check["equivalent"]
            else "repair_failed"
        ),
        "counterexample_count": len(counterexamples),
        "counterexamples": counterexamples,
        "original_verification": initial_check,
        "repaired_qubo": repaired_qubo,
        "final_verification": final_check,
    }
