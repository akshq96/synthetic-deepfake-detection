"""Launches ml.training.train as a subprocess and tracks it via a `Run` DB
row + PID — the intentionally simple alternative to a job queue (Celery/
Redis) the project plan calls for: this is a solo/local-first tool with at
most one training job meaningfully running at a time (one GPU/CPU), so a
queue would add infra with no corresponding benefit.

Known limitation, accepted rather than engineered around: `_ACTIVE_PROCESSES`
lives in this process's memory, so if the FastAPI server restarts while a
job is running, its exit code can no longer be observed — polling falls back
to `psutil.pid_exists`, and a since-exited process is reported failed with an
explanatory message rather than silently stuck "running" forever.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy.orm import Session

from backend.app.core.config import settings
from backend.app.models.experiment import Experiment
from backend.app.models.run import Run
from ml.training.config import REPO_ROOT

_ACTIVE_PROCESSES: dict[str, subprocess.Popen] = {}


def _get_or_create_experiment(db: Session, name: str) -> Experiment:
    experiment = db.query(Experiment).filter(Experiment.name == name).one_or_none()
    if experiment is None:
        experiment = Experiment(name=name)
        db.add(experiment)
        db.commit()
        db.refresh(experiment)
    return experiment


def launch_training_job(
    db: Session,
    *,
    experiment_name: str,
    run_name: str,
    config_path: str,
    overrides: list[str] | None = None,
) -> Run:
    experiment = _get_or_create_experiment(db, experiment_name)
    # Force the subprocess to write into this backend's own artifacts_root
    # (checkpoints, MLflow store) rather than whatever relative default the
    # config file happens to have — otherwise a redirected artifacts_root
    # (e.g. in tests, or a non-default deployment) silently diverges from
    # where the backend itself looks for results.
    overrides = [f"artifacts_root={settings.artifacts_root}", *(overrides or [])]

    run = Run(
        experiment_id=experiment.id,
        run_name=run_name,
        config_snapshot=json.dumps({"config_path": config_path, "overrides": overrides}),
        status="pending",
    )
    db.add(run)
    db.commit()
    db.refresh(run)

    log_dir = settings.experiments_dir / run.id
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "log.txt"

    command = [
        sys.executable,
        "-m",
        "ml.training.train",
        "--config",
        config_path,
        f"run_name={run_name}",
        *overrides,
    ]

    log_file = open(log_path, "w")  # noqa: SIM115 — kept open for the subprocess's lifetime, closed on completion poll
    process = subprocess.Popen(
        command,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        cwd=str(REPO_ROOT),
    )
    _ACTIVE_PROCESSES[run.id] = process

    run.status = "running"
    run.pid = process.pid
    run.log_path = str(log_path)
    run.started_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(run)
    return run


def _extract_mlflow_run_id(log_path: str) -> str | None:
    try:
        text = Path(log_path).read_text()
    except OSError:
        return None
    for line in text.splitlines():
        if "mlflow_run_id=" in line:
            return line.split("mlflow_run_id=")[1].split()[0]
    return None


def poll_run_status(db: Session, run: Run) -> Run:
    """Refreshes and returns `run`'s status. Cheap to call repeatedly (e.g.
    from a frontend polling GET /api/experiments/{id}/status)."""
    if run.status != "running":
        return run

    process = _ACTIVE_PROCESSES.get(run.id)
    if process is not None:
        returncode = process.poll()
        if returncode is None:
            return run  # still running
        run.finished_at = datetime.now(timezone.utc)
        run.status = "completed" if returncode == 0 else "failed"
        if returncode != 0:
            run.error_message = f"training subprocess exited with code {returncode}; see {run.log_path}"
        run.mlflow_run_id = _extract_mlflow_run_id(run.log_path) if run.log_path else None
        del _ACTIVE_PROCESSES[run.id]
        db.commit()
        db.refresh(run)
        return run

    # No in-memory handle (server restarted since launch) — fall back to
    # checking whether the PID is still alive.
    import psutil

    if run.pid and psutil.pid_exists(run.pid):
        return run  # presumed still running; can't distinguish from a reused PID at this scope
    run.status = "failed"
    run.finished_at = datetime.now(timezone.utc)
    run.error_message = (
        "process is no longer running and its exit status could not be determined "
        "(the backend was restarted after this job was launched)"
    )
    run.mlflow_run_id = _extract_mlflow_run_id(run.log_path) if run.log_path else None
    db.commit()
    db.refresh(run)
    return run
