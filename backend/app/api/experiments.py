"""Experiment/run listing + the Synthetic Data Lab's "launch a run" endpoint.

Note on `{id}` path params: `/api/experiments` and `/api/experiments/{id}`
address `Experiment` rows (a named group of runs); `/api/experiments/{id}/status`
addresses a single `Run` row's id (the id returned by the synthetic-lab
launch endpoint) — a run's status is what's actually being polled, per the
project plan's endpoint list.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session, selectinload

from backend.app.db.base import get_db
from backend.app.models.experiment import Experiment
from backend.app.models.run import Run
from backend.app.schemas.experiment import ExperimentOut, RunOut, RunStatusOut, SyntheticLabRunRequest
from backend.app.services.job_launcher import launch_training_job, poll_run_status

router = APIRouter(prefix="/api/experiments", tags=["experiments"])


@router.get("", response_model=list[ExperimentOut])
def list_experiments(db: Session = Depends(get_db)) -> list[ExperimentOut]:
    experiments = (
        db.query(Experiment).options(selectinload(Experiment.runs)).order_by(Experiment.created_at.desc()).all()
    )
    return [ExperimentOut.model_validate(e) for e in experiments]


@router.get("/{experiment_id}", response_model=ExperimentOut)
def get_experiment(experiment_id: str, db: Session = Depends(get_db)) -> ExperimentOut:
    experiment = (
        db.query(Experiment)
        .options(selectinload(Experiment.runs))
        .filter(Experiment.id == experiment_id)
        .one_or_none()
    )
    if experiment is None:
        raise HTTPException(status_code=404, detail=f"experiment {experiment_id!r} not found")
    return ExperimentOut.model_validate(experiment)


@router.post("/synthetic-lab/run", response_model=RunOut)
def run_synthetic_lab_experiment(payload: SyntheticLabRunRequest, db: Session = Depends(get_db)) -> RunOut:
    base_config_path = Path(payload.base_config_path)
    if not base_config_path.exists():
        raise HTTPException(status_code=400, detail=f"base config not found: {base_config_path}")

    overrides = [
        f"data.synthetic_ratio={payload.synthetic_ratio}",
        f"data.synthetic_techniques=[{','.join(payload.synthetic_techniques)}]",
    ]
    if payload.model_name:
        overrides.append(f"model.name={payload.model_name}")
    if payload.manifest_path:
        overrides.append(f"data.manifest_path={payload.manifest_path}")
    if payload.epochs:
        overrides.append(f"training.epochs={payload.epochs}")

    run = launch_training_job(
        db,
        experiment_name=payload.experiment_name,
        run_name=payload.run_name,
        config_path=str(base_config_path),
        overrides=overrides,
    )
    return RunOut.model_validate(run)


@router.get("/{run_id}/status", response_model=RunStatusOut)
def get_run_status(run_id: str, db: Session = Depends(get_db)) -> RunStatusOut:
    run = db.get(Run, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"run {run_id!r} not found")
    run = poll_run_status(db, run)
    return RunStatusOut(
        id=run.id, status=run.status, pid=run.pid, error_message=run.error_message, mlflow_run_id=run.mlflow_run_id
    )
