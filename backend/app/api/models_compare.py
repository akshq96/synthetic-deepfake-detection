"""CNN vs ViT comparison view data — the most recent MLflow run per
architecture (by `model.name` param) within a given experiment, so the
frontend can show both models' metrics side by side.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from backend.app.core.config import settings

router = APIRouter(prefix="/api/models", tags=["models"])


@router.get("/compare")
def compare_models(experiment_name: str = Query(...)) -> dict:
    import mlflow

    client = mlflow.tracking.MlflowClient(tracking_uri=settings.resolved_mlflow_tracking_uri)
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise HTTPException(status_code=404, detail=f"MLflow experiment {experiment_name!r} not found")

    runs = client.search_runs([experiment.experiment_id], order_by=["start_time DESC"])

    by_model: dict[str, dict] = {}
    for run in runs:
        model_name = run.data.params.get("model.name")
        if model_name and model_name not in by_model:
            by_model[model_name] = {
                "run_id": run.info.run_id,
                "run_name": run.data.tags.get("mlflow.runName"),
                "params": dict(run.data.params),
                "metrics": dict(run.data.metrics),
            }

    return {"experiment_name": experiment_name, "models": by_model}
