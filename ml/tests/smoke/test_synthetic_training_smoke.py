"""Phase 3 smoke test: a training run with synthetic_ratio=0.3 completes end
to end on fixture data and the ratio is logged as an MLflow param — the
"definition of done" for Phase 3 per the project plan.
"""

from __future__ import annotations

from pathlib import Path

from ml.data_pipeline.leakage_check import check_no_leakage
from ml.data_pipeline.manifest import load_manifest, save_manifest
from ml.data_pipeline.schema import SPLIT_TRAIN
from ml.data_pipeline.split import assign_splits
from ml.tests.fixtures.synthetic_fixture import generate_fixture_dataset
from ml.training.config import load_config
from ml.training.train import run_training


def test_synthetic_augmented_training_smoke(tmp_path: Path):
    raw = generate_fixture_dataset(tmp_path / "raw")
    split = assign_splits(raw, seed=42)
    manifest_path = tmp_path / "manifests" / "fixture_split.parquet"
    save_manifest(split, manifest_path)

    artifacts_root = tmp_path / "artifacts"
    cfg = load_config(
        Path("ml/configs/cnn_synthetic_debug.yaml"),
        cli_overrides=[
            f"data.manifest_path={manifest_path}",
            f"artifacts_root={artifacts_root}",
            "run_name=cnn_synthetic_debug_smoke",
        ],
    )

    summary = run_training(cfg)

    assert Path(summary["checkpoint_path"]).exists()
    assert summary["val_metrics"] is not None

    # Synthetic images should have actually been written under artifacts_root.
    synthetic_dir = artifacts_root / "synthetic" / "cnn_synthetic_debug_smoke"
    assert synthetic_dir.exists()
    assert len(list(synthetic_dir.glob("*.jpg"))) > 0

    import mlflow

    client = mlflow.tracking.MlflowClient(tracking_uri=f"sqlite:///{artifacts_root / 'mlflow.db'}")
    run = client.get_run(summary["mlflow_run_id"])
    assert run.data.params.get("data.synthetic_ratio") == "0.3"
    assert "blend_warp" in run.data.params.get("data.synthetic_techniques", "")


def test_augmented_manifest_still_passes_leakage_check(tmp_path: Path):
    raw = generate_fixture_dataset(tmp_path / "raw")
    split = assign_splits(raw, seed=42)
    manifest_path = tmp_path / "manifests" / "fixture_split.parquet"
    save_manifest(split, manifest_path)

    from ml.synthetic.mixer import mix_into_manifest

    manifest = load_manifest(manifest_path)
    mixed = mix_into_manifest(
        manifest,
        ratio=0.3,
        techniques=["freq_perturb"],
        output_dir=tmp_path / "synthetic",
        seed=42,
    )
    check_no_leakage(mixed)
    assert (mixed["split"] == SPLIT_TRAIN).sum() > (manifest["split"] == SPLIT_TRAIN).sum()
