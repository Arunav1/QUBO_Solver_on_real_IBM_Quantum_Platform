
from math import isfinite
from typing import Any

import numpy as np

from .constraints import normalize_constraint


def evaluate_compiled_qubo(
    qubo: dict[str, Any],
    assignment: dict[str, int],
) -> float:
    """Evaluate the QUBO returned by the compiler."""
    energy = float(qubo.get("constant", 0.0))

    for term in qubo["terms"]:
        energy += (
            float(term["coefficient"])
            * assignment[term["var_i"]]
            * assignment[term["var_j"]]
        )

    return energy


def evaluate_expected_energy(
    model,
    assignment: dict[str, int],
    penalty: float,
) -> float:
    """
    Independently evaluate objective plus constraint-violation penalties.

    Each constraint is put in its exact integer form first, then

    Equality:      violation = lhs - rhs
    Inequality:    violation = max(0, lhs - rhs)   (normalized to <=)

    and the energy adds penalty * violation**2.
    """
    objective = sum(
        float(coefficient) * assignment[variable]
        for variable, coefficient in model.objective.items()
    )

    if model.sense == "min":
        energy = objective
    elif model.sense == "max":
        energy = -objective
    else:
        raise ValueError(f"Unsupported objective sense: {model.sense}")

    for constraint in model.constraints:
        form = normalize_constraint(constraint)
        violation = form.violation(form.lhs(assignment))
        energy += penalty * float(violation) ** 2

    return float(energy)


def _bit_columns(count: int, total_bits: int) -> list[np.ndarray]:
    """Bit k of every integer in [0, 2**total_bits) for the first `count` bits."""
    states = np.arange(2**total_bits, dtype=np.int64)
    return [((states >> k) & 1).astype(np.int8) for k in range(count)]


def _expected_energies(model, variables, penalty) -> np.ndarray:
    """Vectorized evaluate_expected_energy over all decision assignments."""
    n = len(variables)
    bits = dict(zip(variables, _bit_columns(n, n)))
    energies = np.zeros(2**n, dtype=float)

    sign = 1.0 if model.sense == "min" else -1.0
    for variable, coefficient in model.objective.items():
        energies += sign * float(coefficient) * bits[variable]

    for constraint in model.constraints:
        form = normalize_constraint(constraint)
        lhs = np.zeros(2**n, dtype=np.int64)

        for variable, a in form.coefficients.items():
            lhs += a * bits[variable].astype(np.int64)

        violation = lhs - form.rhs
        if form.sense == "<=":
            violation = np.maximum(violation, 0)

        energies += penalty * violation.astype(float) ** 2

    return energies


def _compiled_energies(qubo, order) -> np.ndarray:
    """Evaluate the QUBO on all 2**len(order) assignments."""
    total = len(order)
    columns = dict(zip(order, _bit_columns(total, total)))
    energies = np.full(2**total, float(qubo.get("constant", 0.0)))

    for term in qubo.get("terms", []):
        coefficient = float(term["coefficient"])
        product = columns[term["var_i"]] & columns[term["var_j"]]
        energies += coefficient * product

    return energies


def verify_qubo_equivalence(
    model,
    qubo: dict[str, Any],
    penalty: float,
    max_variables: int = 20,
    tolerance: float = 1e-8,
) -> dict:
    """
    Exhaustively compare the compiled QUBO with the independently
    reconstructed objective-plus-penalty energy.

    QUBO variables that are not in the model are treated as auxiliary
    (e.g. inequality slack). For every decision assignment x the check
    requires

        min over auxiliary bits s of QUBO(x, s) == expected_energy(x)

    which is exactly what makes the QUBO minimizer, restricted to the
    decision variables, the penalized optimum of the original model.

    This proves equivalence over all binary assignments only for
    the supported model class and the supplied penalty.
    """
    variables = list(model.variables)

    if not variables:
        raise ValueError("The model must contain at least one variable.")

    if len(set(variables)) != len(variables):
        raise ValueError("Model variable names must be unique.")

    if not isfinite(penalty) or penalty <= 0:
        raise ValueError("Penalty must be a finite positive number.")

    if not isfinite(tolerance) or tolerance < 0:
        raise ValueError("Tolerance must be finite and non-negative.")

    qubo_variables = list(qubo.get("variables", []))

    if len(set(qubo_variables)) != len(qubo_variables):
        raise ValueError("QUBO variable names must be unique.")

    missing = [v for v in variables if v not in qubo_variables]
    if missing:
        return {
            "equivalent": False,
            "error": "QUBO is missing model variables.",
            "missing_variables": missing,
            "model_variables": variables,
            "qubo_variables": qubo_variables,
        }

    auxiliary = [v for v in qubo_variables if v not in set(variables)]
    order = variables + auxiliary

    if len(order) > max_variables:
        raise ValueError(
            f"Exact equivalence verification supports at most "
            f"{max_variables} QUBO variables; received {len(order)} "
            f"({len(variables)} decision + {len(auxiliary)} slack)."
        )

    # Check model references before enumeration.
    for variable in model.objective:
        if variable not in variables:
            raise ValueError(
                f"Objective references unknown variable: {variable}"
            )

    for constraint in model.constraints:
        for variable in constraint.coefficients:
            if variable not in variables:
                raise ValueError(
                    f"Constraint '{constraint.name}' references "
                    f"unknown variable: {variable}"
                )

    for term in qubo.get("terms", []):
        if term["var_i"] not in order or term["var_j"] not in order:
            raise ValueError(f"QUBO term references an unknown variable: {term}")

        coefficient = float(term["coefficient"])
        if not isfinite(coefficient):
            raise ValueError("QUBO coefficients must be finite.")

    constant = float(qubo.get("constant", 0.0))
    if not isfinite(constant):
        raise ValueError("QUBO constant must be finite.")

    n = len(variables)
    m = len(auxiliary)

    # Decision bits are the low bits, so row s / column x of the
    # reshaped array holds QUBO(x, s).
    compiled = _compiled_energies(qubo, order).reshape(2**m, 2**n)
    effective = compiled.min(axis=0)
    expected = _expected_energies(model, variables, penalty)

    if not np.all(np.isfinite(expected)) or not np.all(np.isfinite(effective)):
        raise ValueError("Encountered a non-finite energy.")

    errors = np.abs(expected - effective)
    # Scale the tolerance with the energy so large penalties do not
    # trigger false mismatches from floating-point rounding.
    allowed = tolerance * np.maximum(1.0, np.abs(expected))
    failing = np.flatnonzero(errors > allowed)

    mismatches = []
    for index in failing[:20]:
        index = int(index)
        mismatches.append({
            "assignment": {
                variable: (index >> k) & 1
                for k, variable in enumerate(variables)
            },
            "expected_energy": float(expected[index]),
            "compiled_energy": float(effective[index]),
            "absolute_error": float(errors[index]),
        })

    return {
        "equivalent": len(failing) == 0,
        "assignments_checked": 2 ** (n + m),
        "decision_assignments_checked": 2**n,
        "auxiliary_variables": auxiliary,
        "max_absolute_error": float(errors.max()),
        "tolerance": tolerance,
        "penalty": penalty,
        "mismatches": mismatches,
        "mismatches_truncated": len(failing) > 20,
    }
