
"""
On-disk records of submitted IBM Quantum jobs.

A hardware job can sit in the queue for a long time, so everything
needed to finish the pipeline (model, verified QUBO, QAOA metadata)
is saved at submission and survives server restarts.
"""

import json
import os
import re
from pathlib import Path


JOB_DIR = Path(
    os.getenv(
        "IBM_JOB_DIR",
        Path(__file__).resolve().parent.parent / "ibm_jobs",
    )
)

_JOB_ID = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


def _path(job_id: str) -> Path:
    if not _JOB_ID.match(job_id):
        raise ValueError("Invalid job id.")
    return JOB_DIR / f"{job_id}.json"


def save_job(job_id: str, record: dict) -> None:
    JOB_DIR.mkdir(parents=True, exist_ok=True)
    path = _path(job_id)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record), encoding="utf-8")
    temporary.replace(path)


def load_job(job_id: str) -> dict | None:
    path = _path(job_id)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def update_job(job_id: str, **fields) -> None:
    record = load_job(job_id)
    if record is None:
        raise ValueError(f"Unknown job: {job_id}")
    record.update(fields)
    save_job(job_id, record)
