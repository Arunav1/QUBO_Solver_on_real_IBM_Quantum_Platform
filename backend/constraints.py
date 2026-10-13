
from dataclasses import dataclass
from math import ceil, floor, gcd


MAX_DECIMALS = 6


@dataclass(frozen=True)
class NormalizedConstraint:
    """
    Integer form of a constraint over binary variables.

    sense "==": sum(a_i * x_i) == rhs
    sense "<=": sum(a_i * x_i) <= rhs

    Every coefficient and the RHS are integers, so any violation is a
    nonzero integer. That makes penalty * violation**2 >= penalty for
    every infeasible assignment.
    """

    name: str
    sense: str
    coefficients: dict[str, int]
    rhs: int

    @property
    def min_lhs(self) -> int:
        return sum(a for a in self.coefficients.values() if a < 0)

    @property
    def max_lhs(self) -> int:
        return sum(a for a in self.coefficients.values() if a > 0)

    def lhs(self, assignment) -> int:
        return sum(
            a * assignment[variable]
            for variable, a in self.coefficients.items()
        )

    def violation(self, lhs):
        """Penalty base: lhs - rhs for ==, max(0, lhs - rhs) for <=."""
        if self.sense == "==":
            return lhs - self.rhs
        return max(0, lhs - self.rhs)


def _is_integer(value: float) -> bool:
    return abs(value - round(value)) <= 1e-9 * max(1.0, abs(value))


def _integer_scale(values, name: str) -> int:
    """Smallest power of ten that turns every value into an integer."""
    for decimals in range(MAX_DECIMALS + 1):
        scale = 10**decimals
        if all(_is_integer(value * scale) for value in values):
            return scale

    raise ValueError(
        f"Constraint '{name}' uses more than {MAX_DECIMALS} decimal "
        "places; round its coefficients before compiling."
    )


def normalize_constraint(constraint) -> NormalizedConstraint:
    """
    Convert ==, <=, >=, <, > into an equivalent integer == or <= form.

    Over binary variables the left-hand side only takes values on the
    grid defined by its coefficients, so strict inequalities and
    non-integer right-hand sides can be tightened exactly.
    """
    name = constraint.name
    sense = constraint.sense
    rhs = float(constraint.rhs)
    raw = {
        variable: float(a)
        for variable, a in constraint.coefficients.items()
        if float(a) != 0.0
    }

    if sense == "==":
        scale = _integer_scale([*raw.values(), rhs], name)
        coefficients = {v: round(a * scale) for v, a in raw.items()}
        bound = round(rhs * scale)

        divisor = gcd(*coefficients.values(), bound) if coefficients else 0
        if divisor > 1:
            coefficients = {v: a // divisor for v, a in coefficients.items()}
            bound //= divisor

        return NormalizedConstraint(name, "==", coefficients, bound)

    scale = _integer_scale(list(raw.values()), name)
    coefficients = {v: round(a * scale) for v, a in raw.items()}
    scaled_rhs = rhs * scale

    if _is_integer(scaled_rhs):
        scaled_rhs = round(scaled_rhs)
        lower, upper = scaled_rhs, scaled_rhs
    else:
        lower, upper = floor(scaled_rhs), ceil(scaled_rhs)

    # Express everything as sum(a_i * x_i) <= bound.
    if sense == "<=":
        bound = lower
    elif sense == "<":
        bound = upper - 1
    elif sense == ">=":
        coefficients = {v: -a for v, a in coefficients.items()}
        bound = -upper
    elif sense == ">":
        coefficients = {v: -a for v, a in coefficients.items()}
        bound = -(lower + 1)
    else:
        raise ValueError(f"Unsupported constraint sense: {sense}")

    # The LHS is always a multiple of the coefficient GCD, so the bound
    # can be rounded down to that multiple without changing feasibility.
    divisor = gcd(*coefficients.values()) if coefficients else 0
    if divisor > 1:
        coefficients = {v: a // divisor for v, a in coefficients.items()}
        bound = bound // divisor

    return NormalizedConstraint(name, "<=", coefficients, bound)


def suggest_penalty(model) -> float:
    """
    Smallest safe integer penalty.

    Normalized violations are integers, so any infeasible assignment
    pays at least `penalty`. Exceeding the objective's full range
    guarantees it can never beat a feasible assignment.
    """
    objective_range = sum(abs(float(c)) for c in model.objective.values())
    return float(floor(objective_range) + 1)
