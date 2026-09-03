"""Phase 4 smoke test: proves ml/training/train.py trains a ViT exactly the
same way it trains a CNN (Phase 2) — architecture is purely a config choice,
per ml/models/factory.py. Definition of done for Phase 4 per the project plan.
"""

from __future__ import annotations

from pathlib import Path

from ml.data_pipeline.manifest import save_manifest
from ml.data_pipeline.split import assign_splits
from ml.tests.fixtures.synthetic_fixture import generate_fixture_dataset
from ml.training.config import load_config
from ml.training.train import run_training


def test_vit_training_smoke(tmp_path: Path):
    raw = generate_fixture_dataset(tmp_path / "raw")
    split = assign_splits(raw, seed=42)
    manifest_path = tmp_path / "manifests" / "fixture_split.parquet"
    save_manifest(split, manifest_path)

    artifacts_root = tmp_path / "artifacts"
    cfg = load_config(
        Path("ml/configs/vit_baseline_debug.yaml"),
        cli_overrides=[
            f"data.manifest_path={manifest_path}",
            f"artifacts_root={artifacts_root}",
            "run_name=vit_baseline_debug_smoke",
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
    assert run.data.params.get("model.name") == "vit_base_patch16_224"
