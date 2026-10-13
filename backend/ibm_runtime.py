
"""
Sampling on IBM Quantum hardware through Qiskit Runtime.

QAOA angles are optimized classically (exact simulation); only the
final optimized circuit is sampled on the QPU. That keeps each solve
to a single hardware job, which matters on the Open plan's monthly
QPU allowance and avoids queueing once per optimizer iteration.

Jobs are asynchronous: submit_sampler_job returns immediately and
fetch_job_counts reports progress until the counts are available.
"""

import os
from functools import lru_cache

from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2


DEFAULT_CHANNEL = "ibm_quantum_platform"
DEFAULT_BACKEND = "least_busy"
AUTO_BACKENDS = {"", "auto", "least_busy"}

PENDING_STATES = {"INITIALIZING", "QUEUED", "VALIDATING", "RUNNING"}
FAILED_STATES = {"ERROR", "CANCELLED"}


def ibm_configured() -> bool:
    return bool(os.getenv("IBM_QUANTUM_API_KEY"))


def backend_setting() -> str:
    """
    IBM_QUANTUM_BACKEND may be one backend name, a comma-separated list
    of candidates, or "least_busy" for every backend the account can use.
    """
    return os.getenv("IBM_QUANTUM_BACKEND", DEFAULT_BACKEND).strip()


def select_backend(service: QiskitRuntimeService, num_qubits: int):
    """
    Pick the configured backend, or the candidate with the shortest
    queue. On the shared Open plan the queue dominates turnaround time,
    so routing to the least busy device is the biggest latency win.
    """
    setting = backend_setting()

    if setting.lower() in AUTO_BACKENDS:
        candidates = service.backends(
            operational=True,
            simulator=False,
            min_num_qubits=num_qubits,
        )
    else:
        names = [name.strip() for name in setting.split(",") if name.strip()]
        candidates = [service.backend(name) for name in names]

    ranked = []
    for backend in candidates:
        status = backend.status()
        if status.operational and backend.num_qubits >= num_qubits:
            ranked.append((status.pending_jobs, backend, status))

    if not ranked:
        raise ValueError(
            f"No operational IBM backend with at least {num_qubits} "
            f"qubits is available (configured: {setting or 'least_busy'})."
        )

    ranked.sort(key=lambda item: item[0])
    _, backend, status = ranked[0]
    return backend, status


@lru_cache(maxsize=1)
def get_service() -> QiskitRuntimeService:
    api_key = os.getenv("IBM_QUANTUM_API_KEY")

    if not api_key:
        raise ValueError(
            "IBM_QUANTUM_API_KEY is not configured. "
            "Set it in .env to run on IBM Quantum hardware."
        )

    options = {
        "channel": os.getenv("IBM_QUANTUM_CHANNEL", DEFAULT_CHANNEL),
        "token": api_key,
    }

    instance = os.getenv("IBM_QUANTUM_INSTANCE")
    if instance:
        options["instance"] = instance

    return QiskitRuntimeService(**options)


def _status_name(job) -> str:
    status = job.status()
    return str(getattr(status, "name", status)).upper()


def submit_sampler_job(circuit, shots: int) -> dict:
    """Transpile for the configured backend and submit one Sampler job."""
    backend, status = select_backend(get_service(), circuit.num_qubits)

    pass_manager = generate_preset_pass_manager(
        optimization_level=3,
        backend=backend,
    )
    isa_circuit = pass_manager.run(circuit)

    sampler = SamplerV2(mode=backend)
    sampler.options.default_shots = shots
    sampler.options.dynamical_decoupling.enable = True
    sampler.options.dynamical_decoupling.sequence_type = "XY4"

    job = sampler.run([isa_circuit])

    return {
        "job_id": job.job_id(),
        "backend": backend.name,
        "shots": shots,
        "pending_jobs_at_submit": status.pending_jobs,
        "transpiled_depth": isa_circuit.depth(),
        "two_qubit_gates": isa_circuit.count_ops().get("cz", 0),
    }


def fetch_job_counts(job_id: str) -> dict:
    """
    Report a job's state. Once it has finished, include the counts
    of the measured bitstrings (Qiskit ordering, qubit 0 rightmost).
    """
    job = get_service().job(job_id)
    state = _status_name(job)

    if state in PENDING_STATES:
        return {"state": state}

    if state in FAILED_STATES:
        message = ""
        try:
            message = job.error_message() or ""
        except Exception:
            pass
        return {"state": state, "error": message or f"IBM job {state.lower()}."}

    if state != "DONE":
        return {"state": state}

    result = job.result()
    counts = result[0].data.meas.get_counts()

    quantum_seconds = None
    try:
        quantum_seconds = job.usage()
    except Exception:
        pass

    return {
        "state": state,
        "counts": counts,
        "quantum_seconds": quantum_seconds,
    }
