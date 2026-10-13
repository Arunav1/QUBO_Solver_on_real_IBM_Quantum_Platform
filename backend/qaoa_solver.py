
from collections import Counter
from typing import Any

import numpy as np
from scipy.optimize import minimize

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def qubo_energy(qubo: dict[str, Any], assignment: dict[str, int]) -> float:
    """Evaluate the original QUBO polynomial."""
    energy = float(qubo.get("constant", 0.0))

    for term in qubo["terms"]:
        energy += (
            float(term["coefficient"])
            * assignment[term["var_i"]]
            * assignment[term["var_j"]]
        )

    return energy


def qubo_to_ising(qubo: dict[str, Any]):
    """
    Convert E(x) into an equivalent Ising energy:

        E(x) = offset + sum(h_i * z_i) + sum(J_ij * z_i*z_j)

    where x_i = (1 - z_i) / 2 and z_i is in {-1, +1}.
    """
    variables = list(qubo["variables"])
    n = len(variables)

    if len(set(variables)) != n:
        raise ValueError("QUBO variable names must be unique.")

    index = {name: i for i, name in enumerate(variables)}
    h = np.zeros(n, dtype=float)
    couplings = {}
    offset = float(qubo.get("constant", 0.0))

    for term in qubo["terms"]:
        vi = term["var_i"]
        vj = term["var_j"]
        coefficient = float(term["coefficient"])

        if vi not in index or vj not in index:
            raise ValueError(f"Unknown QUBO variable in term: {term}")

        i, j = index[vi], index[vj]

        if i == j:
            # x_i^2 = x_i for binary variables.
            offset += coefficient / 2.0
            h[i] -= coefficient / 2.0
        else:
            # a*x_i*x_j = a/4 * (1 - Z_i - Z_j + Z_i*Z_j)
            offset += coefficient / 4.0
            h[i] -= coefficient / 4.0
            h[j] -= coefficient / 4.0

            key = (min(i, j), max(i, j))
            couplings[key] = couplings.get(key, 0.0) + coefficient / 4.0

    return variables, offset, h, couplings


def build_qaoa_circuit(
    h: np.ndarray,
    couplings: dict,
    gammas: np.ndarray,
    betas: np.ndarray,
) -> QuantumCircuit:
    """Build a standard QAOA circuit for the Ising cost Hamiltonian."""
    n = len(h)
    p = len(gammas)

    circuit = QuantumCircuit(n)

    # Uniform superposition over all binary assignments.
    circuit.h(range(n))

    for layer in range(p):
        gamma = float(gammas[layer])
        beta = float(betas[layer])

        # Cost unitary: exp(-i * gamma * H_cost).
        for i, coefficient in enumerate(h):
            if abs(coefficient) > 1e-12:
                circuit.rz(2.0 * gamma * coefficient, i)

        for (i, j), coefficient in couplings.items():
            if abs(coefficient) > 1e-12:
                circuit.rzz(2.0 * gamma * coefficient, i, j)

        # Mixer unitary: exp(-i * beta * sum(X_i)).
        circuit.rx(2.0 * beta, range(n))

    return circuit


def basis_energies(qubo: dict[str, Any], variables: list[str]) -> np.ndarray:
    """
    QUBO energy of every computational basis state.

    Index bit i (qubit i, least significant first) is variable i.
    """
    n = len(variables)
    states = np.arange(2**n, dtype=np.int64)
    bits = {
        variable: ((states >> i) & 1).astype(np.int8)
        for i, variable in enumerate(variables)
    }
    energies = np.full(2**n, float(qubo.get("constant", 0.0)))

    for term in qubo["terms"]:
        energies += float(term["coefficient"]) * (
            bits[term["var_i"]] & bits[term["var_j"]]
        )

    return energies


def qaoa_probabilities(
    energies: np.ndarray,
    n: int,
    gammas: np.ndarray,
    betas: np.ndarray,
) -> np.ndarray:
    """
    Exact output distribution of build_qaoa_circuit.

    The cost unitary is diagonal, exp(-i * gamma * E(x)) up to a global
    phase, and the mixer is RX(2*beta) on every qubit, so the state can
    be propagated directly without building a circuit per iteration.
    """
    state = np.full(2**n, 2 ** (-n / 2), dtype=complex)

    for gamma, beta in zip(gammas, betas):
        state *= np.exp(-1j * float(gamma) * energies)

        cos, sin = np.cos(float(beta)), -1j * np.sin(float(beta))
        for qubit in range(n):
            view = state.reshape(-1, 2, 2**qubit)
            zero = view[:, 0, :].copy()
            one = view[:, 1, :]
            view[:, 0, :] = cos * zero + sin * one
            view[:, 1, :] = sin * zero + cos * one

    return np.abs(state) ** 2


def group_samples(
    candidates: list[dict],
    decision_variables: list[str],
    variables: list[str],
    energies: np.ndarray,
) -> tuple[list[dict], dict]:
    """
    Merge samples that agree on the decision variables.

    Auxiliary (slack) bits carry no decision meaning: for a fixed
    decision assignment the best slack setting is the one minimizing
    the QUBO. Each sampled decision assignment is therefore scored with
    that minimum ("energy"), alongside the lowest energy actually
    measured for it ("sampled_energy"). Only assignments QAOA sampled
    are considered.

    Returns the grouped samples (lowest energy first) and the best
    sample with its full assignment, including optimal slack bits.
    """
    positions = [variables.index(v) for v in decision_variables]
    states = np.arange(len(energies), dtype=np.int64)
    keys = np.zeros(len(energies), dtype=np.int64)

    for k, position in enumerate(positions):
        keys |= ((states >> position) & 1) << k

    effective = np.full(2 ** len(positions), np.inf)
    np.minimum.at(effective, keys, energies)

    groups = {}

    for sample in candidates:
        bits = tuple(sample["assignment"][v] for v in decision_variables)
        group = groups.get(bits)

        if group is None:
            key = sum(bit << k for k, bit in enumerate(bits))
            groups[bits] = {
                "assignment": dict(zip(decision_variables, bits)),
                "energy": float(effective[key]),
                "sampled_energy": sample["energy"],
                "count": sample["count"],
                "probability": sample["probability"],
                "_key": key,
            }
            continue

        group["count"] += sample["count"]
        group["probability"] += sample["probability"]
        group["sampled_energy"] = min(
            group["sampled_energy"], sample["energy"]
        )

    samples = sorted(groups.values(), key=lambda item: item["energy"])
    top = samples[0]

    matching = np.flatnonzero(keys == top["_key"])
    best_state = int(matching[np.argmin(energies[matching])])

    for group in samples:
        del group["_key"]

    best = {
        "assignment": {
            variable: (best_state >> i) & 1
            for i, variable in enumerate(variables)
        },
        "decision_assignment": top["assignment"],
        "energy": top["energy"],
        "sampled_energy": top["sampled_energy"],
        "count": top["count"],
        "probability": top["probability"],
    }

    return samples, best


def decode_bitstring(bitstring: str, variables: list[str]) -> dict[str, int]:
    """
    Qiskit displays classical bitstrings with the highest-index
    bit on the left. Reverse that order to match our variable list.
    """
    bits = bitstring.replace(" ", "")

    if len(bits) != len(variables):
        raise ValueError("Bitstring length does not match QUBO variables.")

    return {
        variable: int(bits[len(bits) - 1 - i])
        for i, variable in enumerate(variables)
    }


def optimize_qaoa(
    qubo: dict[str, Any],
    p: int = 1,
    maxiter: int = 100,
    seed: int = 42,
) -> dict:
    """
    Optimize QAOA angles with exact statevector expectations.

    Returns the optimized circuit (with measurements) ready to be
    sampled on any backend, plus metadata describing the optimization.
    """
    variables, offset, h, couplings = qubo_to_ising(qubo)
    n = len(variables)

    if not 1 <= n <= 20:
        raise ValueError("This implementation supports 1 to 20 variables.")

    if p < 1:
        raise ValueError("p must be at least 1.")

    if maxiter < 1:
        raise ValueError("maxiter must be at least 1.")

    rng = np.random.default_rng(seed)
    initial = np.concatenate([
        rng.uniform(0, np.pi, size=p),
        rng.uniform(0, np.pi / 2, size=p),
    ])

    energies = basis_energies(qubo, variables)

    # Normalize the cost Hamiltonian so gamma in [0, pi] explores a
    # useful range even when large constraint penalties dominate the
    # energy scale. This only rescales gamma; the expectation below
    # is still taken over the true QUBO energies.
    scale = max(
        [abs(value) for value in h] + [abs(value) for value in couplings.values()],
        default=0.0,
    )
    scale = scale if scale > 1e-12 else 1.0
    phase_energies = (energies - offset) / scale

    def expected_energy(parameters):
        probabilities = qaoa_probabilities(
            phase_energies, n, parameters[:p], parameters[p:]
        )
        return float(probabilities @ energies)

    optimization = minimize(
        expected_energy,
        initial,
        method="COBYLA",
        options={"maxiter": maxiter, "rhobeg": 0.5},
    )

    gammas = optimization.x[:p]
    betas = optimization.x[p:]

    circuit = build_qaoa_circuit(
        h / scale,
        {key: value / scale for key, value in couplings.items()},
        gammas,
        betas,
    )
    circuit.measure_all()

    return {
        "circuit": circuit,
        "metadata": {
            "solver": "QAOA",
            "variables": variables,
            "qubits": n,
            "layers": p,
            "optimizer": {
                "method": "COBYLA",
                "success": bool(optimization.success),
                "message": str(optimization.message),
                "iterations": int(optimization.nfev),
            },
            "angles": {
                "gammas": (gammas / scale).tolist(),
                "betas": betas.tolist(),
            },
            "expected_qubo_energy": float(expected_energy(optimization.x)),
            "ising": {
                "offset": float(offset),
                "linear_coefficients": h.tolist(),
                "couplings": [
                    {
                        "var_i": variables[i],
                        "var_j": variables[j],
                        "coefficient": float(value),
                    }
                    for (i, j), value in couplings.items()
                ],
            },
        },
    }


def summarize_counts(
    qubo: dict[str, Any],
    counts: dict[str, int],
    metadata: dict,
    backend: str,
    decision_variables: list[str] | None = None,
) -> dict:
    """Turn measured bitstring counts into the QAOA result payload."""
    variables = metadata["variables"]
    shots = sum(counts.values())

    if shots < 1:
        raise ValueError("The backend returned no measurement shots.")

    candidates = []

    for bitstring, frequency in counts.items():
        assignment = decode_bitstring(bitstring, variables)
        energy = qubo_energy(qubo, assignment)

        candidates.append({
            "assignment": assignment,
            "energy": energy,
            "count": int(frequency),
            "probability": float(frequency / shots),
        })

    candidates.sort(key=lambda item: item["energy"])
    best = dict(candidates[0])

    samples = candidates
    if decision_variables:
        unknown = set(decision_variables) - set(variables)
        if unknown:
            raise ValueError(f"Unknown decision variables: {sorted(unknown)}")

        samples, best = group_samples(
            candidates,
            list(decision_variables),
            variables,
            basis_energies(qubo, variables),
        )

    return {
        **metadata,
        "backend": backend,
        "shots": shots,
        "best_sample": best,
        "samples": samples,
    }


def solve_qaoa(
    qubo: dict[str, Any],
    p: int = 1,
    shots: int = 2048,
    maxiter: int = 100,
    seed: int = 42,
    decision_variables: list[str] | None = None,
) -> dict:
    """
    Optimize QAOA angles with exact statevector expectations,
    then sample the optimized circuit using Qiskit Aer.

    This is a simulator implementation, not a hardware run; see
    ibm_runtime.py for sampling on IBM Quantum hardware.

    When decision_variables is given, samples are grouped by those
    variables so auxiliary (slack) qubits do not fragment the results.
    """
    if shots < 1:
        raise ValueError("shots must be at least 1.")

    prepared = optimize_qaoa(qubo, p=p, maxiter=maxiter, seed=seed)

    simulator = AerSimulator(seed_simulator=seed)
    compiled = transpile(prepared["circuit"], simulator, optimization_level=1)
    counts = simulator.run(compiled, shots=shots).result().get_counts()

    return summarize_counts(
        qubo,
        counts,
        prepared["metadata"],
        backend="Qiskit Aer simulator",
        decision_variables=decision_variables,
    )
