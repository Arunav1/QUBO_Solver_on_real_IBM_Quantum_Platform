
from itertools import product
from math import ceil, isfinite, log2

from .constraints import normalize_constraint


def compile_qubo(model, penalty=10.0):
    """
    Compile a binary optimization model into QUBO.

    Supports ==, <=, >=, <, > with any coefficient signs and up to
    six decimal places. Each constraint is first normalized to an
    exact integer form (see constraints.normalize_constraint).

    Inequalities use binary-encoded slack variables.
    """

    model.validate_variables()

    if not isfinite(penalty) or penalty <= 0:
        raise ValueError("penalty must be a positive finite number")

    variables = list(model.variables)
    Q = {}
    constant = 0.0

    def add_term(i, j, value):
        if variables.index(i) > variables.index(j):
            i, j = j, i
        Q[(i, j)] = Q.get((i, j), 0.0) + value

    def add_square(coefficients, rhs):
        """Add penalty * (sum(a_i*x_i) - rhs)^2."""
        nonlocal constant

        constant += penalty * rhs**2

        for variable, a in coefficients.items():
            add_term(variable, variable,
                     penalty * (a * a - 2 * rhs * a))

        names = list(coefficients)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                u, v = names[i], names[j]
                add_term(
                    u, v,
                    2 * penalty * coefficients[u] * coefficients[v]
                )

    def add_slack_variables(name, max_slack):
        """Return coefficients for a binary-encoded slack in [0, max_slack]."""
        if max_slack <= 0:
            return {}

        count = ceil(log2(max_slack + 1))
        slack = {}
        weights = [2**i for i in range(count)]

        # Use a bounded final weight so every integer in
        # [0, max_slack] is representable and nothing larger is.
        weights[-1] = max_slack - sum(weights[:-1])

        for index, weight in enumerate(weights):
            var = f"__slack_{name}_{index}"

            if var in variables:
                raise ValueError(f"Reserved variable name: {var}")

            variables.append(var)
            slack[var] = weight

        return slack

    # Convert maximization to minimization.
    sign = 1.0 if model.sense == "min" else -1.0

    for variable, coefficient in model.objective.items():
        add_term(variable, variable, sign * coefficient)

    for index, constraint in enumerate(model.constraints):
        form = normalize_constraint(constraint)
        coeffs = form.coefficients
        rhs = form.rhs

        if form.sense == "==":
            add_square(coeffs, rhs)
            continue

        # sum(a_i*x_i) <= rhs. If even the largest LHS fits, the
        # constraint can never be violated and needs no penalty.
        if form.max_lhs <= rhs:
            continue

        # Encode as sum(a_i*x_i) + slack = rhs with slack in
        # [0, rhs - min_lhs]. When rhs < min_lhs the constraint is
        # infeasible for every assignment and the plain square is
        # already exactly penalty * violation^2.
        slack = add_slack_variables(
            f"{index}_{constraint.name}",
            rhs - form.min_lhs,
        )

        equation = dict(coeffs)
        equation.update(slack)
        add_square(equation, rhs)

    return {
        "variables": variables,
        "constant": constant,
        "terms": [
            {
                "var_i": i,
                "var_j": j,
                "coefficient": coefficient,
            }
            for (i, j), coefficient in Q.items()
            if abs(coefficient) > 1e-12
        ],
    }


def energy(assignment, qubo):
    value = qubo["constant"]

    for term in qubo["terms"]:
        value += (
            term["coefficient"]
            * assignment[term["var_i"]]
            * assignment[term["var_j"]]
        )

    return value


def is_feasible(model, assignment, tol=1e-9):
    """Check the original constraints, not the QUBO penalties."""
    for constraint in model.constraints:
        lhs = sum(
            coefficient * assignment[variable]
            for variable, coefficient in constraint.coefficients.items()
        )
        rhs = constraint.rhs

        if constraint.sense == "==":
            ok = abs(lhs - rhs) <= tol
        elif constraint.sense == "<=":
            ok = lhs <= rhs + tol
        elif constraint.sense == ">=":
            ok = lhs >= rhs - tol
        elif constraint.sense == "<":
            ok = lhs < rhs - tol
        else:  # >
            ok = lhs > rhs + tol

        if not ok:
            return False

    return True


def exact_solve(model, qubo=None):
    """
    Exhaustively solve the ORIGINAL model for up to 20 original variables.
    Slack variables are not included in this enumeration.
    """
    if len(model.variables) > 20:
        raise ValueError("Exact solver supports at most 20 original variables")

    best = None

    for bits in product([0, 1], repeat=len(model.variables)):
        assignment = dict(zip(model.variables, bits))

        if not is_feasible(model, assignment):
            continue

        objective = sum(
            coefficient * assignment[variable]
            for variable, coefficient in model.objective.items()
        )

        if best is None or (
            model.sense == "min" and objective < best["objective"]
        ) or (
            model.sense == "max" and objective > best["objective"]
        ):
            best = {
                "assignment": assignment,
                "objective": objective,
                "energy": energy(assignment, qubo) if qubo else None,
            }

    return best
