"""Detector model registry — enumerates trained, inference-ready checkpoints
for the Detect page's model selector. Distinct from models_compare.py, which
compares training-run *metrics* for the CNN-vs-ViT research page rather than
listing deployable checkpoints.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException

from backend.app.core.config import settings

router = APIRouter(prefix="/api/models", tags=["models"])


def _find_checkpoint_file(run_dir: Path) -> Path | None:
    # best.pt (preferred: overwritten on val-metric improvement) falls back
    # to latest.pt (a run that hasn't hit a "best" checkpoint yet still gets
    # listed) — matches ml/training/checkpoint.py's layout convention.
    for name in ("best.pt", "latest.pt"):
        candidate = run_dir / name
        if candidate.exists():
            return candidate
    return None


PRETRAINED_RUN_ID = "pretrained-vit-deepfake"


def _list_models_raw() -> list[dict]:
    from ml.models.pretrained_detector import MODEL_NAME as PRETRAINED_MODEL_NAME
    from ml.models.pretrained_detector import PRETRAINED_MODEL_ID

    default_checkpoint = settings.default_checkpoint_path
    default_model_name = settings.default_model_name

    # A real, already-trained public model — not one of this project's own
    # checkpoints under artifacts/checkpoints/, so it's synthesized here
    # rather than discovered by scanning the filesystem below.
    models = [
        {
            "run_id": PRETRAINED_RUN_ID,
            "run_name": f"Pretrained ({PRETRAINED_MODEL_ID})",
            "model_name": PRETRAINED_MODEL_NAME,
            "checkpoint_path": "pretrained",
            "is_default": default_model_name == PRETRAINED_MODEL_NAME,
        }
    ]

    checkpoints_root = settings.artifacts_root / "checkpoints"
    if not checkpoints_root.exists():
        return models

    # Cross-reference MLflow for a human-friendly run_name + model
    # architecture, best-effort — a checkpoint dir with no matching MLflow
    # run (e.g. MLflow store was reset) still gets listed under its run_id,
    # not silently dropped.
    run_info: dict[str, dict] = {}
    try:
        import mlflow

        client = mlflow.tracking.MlflowClient(tracking_uri=settings.resolved_mlflow_tracking_uri)
        for experiment in client.search_experiments():
            for run in client.search_runs([experiment.experiment_id], max_results=500):
                run_info[run.info.run_id] = {
                    "run_name": run.data.tags.get("mlflow.runName"),
                    "model_name": run.data.params.get("model.name"),
                }
    except Exception:
        run_info = {}

    for run_dir in sorted(checkpoints_root.iterdir()):
        if not run_dir.is_dir():
            continue
        checkpoint_file = _find_checkpoint_file(run_dir)
        if checkpoint_file is None:
            continue
        run_id = run_dir.name
        info = run_info.get(run_id, {})
        checkpoint_path = str(checkpoint_file)
        models.append(
            {
                "run_id": run_id,
                "run_name": info.get("run_name") or run_id,
                # Falls back to this project's own default architecture, not
                # settings.default_model_name — that setting may now point at
                # the pretrained wrapper above, which is never what a
                # filesystem-discovered checkpoint.pt actually is.
                "model_name": info.get("model_name") or "efficientnetv2_s",
                "checkpoint_path": checkpoint_path,
                "is_default": checkpoint_path == default_checkpoint,
            }
        )
    return models


def resolve_model(model_run_id: str) -> tuple[str, str]:
    """(checkpoint_path, model_name) for a registry entry's run_id — used by
    detect.py to turn the Detect page's model-selector choice into an
    ml_bridge override. Raises HTTPException(404) for an unknown run_id."""
    for entry in _list_models_raw():
        if entry["run_id"] == model_run_id:
            return entry["checkpoint_path"], entry["model_name"]
    raise HTTPException(status_code=404, detail=f"no usable checkpoint found for model_run_id {model_run_id!r}")


@router.get("")
def list_models() -> list[dict]:
    return _list_models_raw()
