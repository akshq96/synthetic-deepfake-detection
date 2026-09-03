"""Thin proxy over the MLflow tracking client for the experiment-history
view — keeps one consistent API surface for the frontend rather than
exposing the raw MLflow UI/API directly.
"""

from __future__ import annotations

from fastapi import APIRouter, Query

from backend.app.core.config import settings

router = APIRouter(prefix="/api/mlflow", tags=["mlflow"])


@router.get("/runs")
def list_mlflow_runs(experiment_name: str | None = Query(None), max_results: int = Query(50, le=500)) -> list[dict]:
    import mlflow

    client = mlflow.tracking.MlflowClient(tracking_uri=settings.resolved_mlflow_tracking_uri)

    if experiment_name:
        experiment = client.get_experiment_by_name(experiment_name)
        experiment_ids = [experiment.experiment_id] if experiment else []
    else:
        experiment_ids = [e.experiment_id for e in client.search_experiments()]

    if not experiment_ids:
        return []

    runs = client.search_runs(experiment_ids, order_by=["start_time DESC"], max_results=max_results)
    return [
        {
            "run_id": r.info.run_id,
            "run_name": r.data.tags.get("mlflow.runName"),
            "experiment_id": r.info.experiment_id,
            "status": r.info.status,
            "start_time": r.info.start_time,
            "params": dict(r.data.params),
            "metrics": dict(r.data.metrics),
        }
        for r in runs
    ]
