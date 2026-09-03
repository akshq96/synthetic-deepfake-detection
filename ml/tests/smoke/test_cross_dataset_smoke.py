"""Phase 5 smoke test: cross-dataset generalization runs end to end — train
on one dataset's train split, evaluate both in-distribution (same dataset's
test split) and cross-dataset (a different dataset's test split), and report
the generalization gap. Definition of done for Phase 5's cross-dataset half.
"""

from __future__ import annotations

from pathlib import Path

from ml.data_pipeline.manifest import merge_manifests, save_manifest
from ml.data_pipeline.split import assign_splits
from ml.evaluation.experiments.cross_dataset import run_cross_dataset_experiment
from ml.tests.fixtures.synthetic_fixture import generate_fixture_dataset
from ml.training.config import load_config


def test_cross_dataset_smoke(tmp_path: Path):
    dataset_a = generate_fixture_dataset(tmp_path / "a", source_dataset="fixtureA", n_identities_per_label=4)
    dataset_b = generate_fixture_dataset(tmp_path / "b", source_dataset="fixtureB", n_identities_per_label=4)
    merged = merge_manifests([dataset_a, dataset_b])
    split = assign_splits(merged, seed=42)

    manifest_path = tmp_path / "manifests" / "merged_split.parquet"
    save_manifest(split, manifest_path)

    artifacts_root = tmp_path / "artifacts"
    cfg = load_config(
        Path("ml/configs/cnn_baseline_debug.yaml"),
        cli_overrides=[
            f"data.manifest_path={manifest_path}",
            f"artifacts_root={artifacts_root}",
            "run_name=cross_dataset_smoke",
            "training.debug_n_samples=40",
        ],
    )

    result = run_cross_dataset_experiment(cfg, train_dataset="fixtureA", eval_dataset="fixtureB")

    assert result["train_dataset"] == "fixtureA"
    assert result["eval_dataset"] == "fixtureB"
    assert result["in_distribution_metrics"] is not None
    assert result["cross_dataset_metrics"] is not None
    assert result["generalization_gap_accuracy"] is not None
    assert Path(result["train_summary"]["checkpoint_path"]).exists()


def test_cross_dataset_unknown_dataset_raises(tmp_path: Path):
    import pytest

    dataset_a = generate_fixture_dataset(tmp_path / "a", source_dataset="fixtureA")
    split = assign_splits(dataset_a, seed=42)
    manifest_path = tmp_path / "manifests" / "a_split.parquet"
    save_manifest(split, manifest_path)

    artifacts_root = tmp_path / "artifacts"
    cfg = load_config(
        Path("ml/configs/cnn_baseline_debug.yaml"),
        cli_overrides=[
            f"data.manifest_path={manifest_path}",
            f"artifacts_root={artifacts_root}",
            "run_name=cross_dataset_bad",
        ],
    )
    with pytest.raises(ValueError, match="not found in manifest"):
        run_cross_dataset_experiment(cfg, train_dataset="fixtureA", eval_dataset="does_not_exist")
