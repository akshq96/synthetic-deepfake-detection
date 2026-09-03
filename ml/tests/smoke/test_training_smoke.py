"""Phase 2 smoke test: extends the Phase 1 dataset-pipeline smoke test through
a full (debug-scale) training run — manifest -> split -> leakage check ->
1-epoch CPU training -> checkpoint -> calibration -> MLflow run. This is the
gate for Phase 2's "definition of done" per the project plan.
"""

from __future__ import annotations

from pathlib import Path

from ml.data_pipeline.manifest import save_manifest
from ml.data_pipeline.split import assign_splits
from ml.tests.fixtures.synthetic_fixture import generate_fixture_dataset
from ml.training.config import load_config
from ml.training.train import run_training


def _prepare_manifest(tmp_path: Path) -> Path:
    raw = generate_fixture_dataset(tmp_path / "raw")
    split = assign_splits(raw, seed=42)
    manifest_path = tmp_path / "manifests" / "fixture_split.parquet"
    save_manifest(split, manifest_path)
    return manifest_path


def test_cnn_training_smoke(tmp_path: Path):
    manifest_path = _prepare_manifest(tmp_path)
    artifacts_root = tmp_path / "artifacts"

    config_path = Path("ml/configs/cnn_baseline_debug.yaml")
    cfg = load_config(
        config_path,
        cli_overrides=[
            f"data.manifest_path={manifest_path}",
            f"artifacts_root={artifacts_root}",
            "run_name=cnn_baseline_debug_smoke",
        ],
    )

    summary = run_training(cfg)

    assert Path(summary["checkpoint_path"]).exists()
    assert Path(summary["calibration_path"]).exists()
    assert summary["val_metrics"] is not None
    assert "accuracy" in summary["val_metrics"]

    import mlflow

    client = mlflow.tracking.MlflowClient(tracking_uri=f"sqlite:///{artifacts_root / 'mlflow.db'}")
    run = client.get_run(summary["mlflow_run_id"])
    assert run.data.params.get("model.name") == "efficientnetv2_s"
    assert "val_accuracy" in run.data.metrics
    assert "calibration_temperature" in run.data.metrics
