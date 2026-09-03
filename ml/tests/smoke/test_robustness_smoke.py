"""Phase 6 smoke test: robustness evaluation runs end to end on a trained
checkpoint and produces degradation curves across perturbation severities.
Definition of done for Phase 6 per the project plan.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ml.data_pipeline.manifest import save_manifest
from ml.data_pipeline.schema import SPLIT_TRAIN
from ml.data_pipeline.split import assign_splits
from ml.evaluation.experiments.robustness import perturb_manifest, run_robustness_experiment
from ml.plotting.robustness_curves import plot_robustness_curve
from ml.synthetic.gaussian_noise import GaussianNoiseTechnique
from ml.tests.fixtures.synthetic_fixture import generate_fixture_dataset
from ml.training.config import load_config
from ml.training.train import run_training


def test_perturb_manifest_preserves_labels_and_splits(tmp_path: Path, fixture_manifest_split: pd.DataFrame):
    subset = fixture_manifest_split[fixture_manifest_split["split"] == SPLIT_TRAIN]
    technique = GaussianNoiseTechnique(sigma=15.0)
    perturbed = perturb_manifest(subset, technique, output_dir=tmp_path / "perturbed", seed=1)

    assert len(perturbed) == len(subset)
    assert list(perturbed["label"].values) == list(subset["label"].values)
    assert list(perturbed["split"].values) == list(subset["split"].values)
    assert list(perturbed["identity_id"].values) == list(subset["identity_id"].values)
    # File paths must actually differ (perturbed copies), and exist on disk.
    assert set(perturbed["file_path"]) != set(subset["file_path"])
    for p in perturbed["file_path"]:
        assert Path(p).exists()


def test_robustness_experiment_smoke(tmp_path: Path):
    raw = generate_fixture_dataset(tmp_path / "raw", n_identities_per_label=4)
    split = assign_splits(raw, seed=42)
    manifest_path = tmp_path / "manifests" / "fixture_split.parquet"
    save_manifest(split, manifest_path)

    artifacts_root = tmp_path / "artifacts"
    cfg = load_config(
        Path("ml/configs/cnn_baseline_debug.yaml"),
        cli_overrides=[
            f"data.manifest_path={manifest_path}",
            f"artifacts_root={artifacts_root}",
            "run_name=robustness_smoke",
            "training.debug_n_samples=40",
        ],
    )
    train_summary = run_training(cfg)

    from ml.data_pipeline.manifest import load_manifest
    from ml.data_pipeline.schema import SPLIT_VAL

    manifest = load_manifest(manifest_path)
    eval_subset = manifest[manifest["split"] == SPLIT_VAL]
    assert len(eval_subset) > 0

    results = run_robustness_experiment(
        cfg,
        checkpoint_path=train_summary["checkpoint_path"],
        calibration_path=train_summary["calibration_path"],
        eval_manifest=eval_subset,
        output_dir=tmp_path / "robustness_out",
        perturbations={"gaussian_noise": [5.0, 30.0], "compression": [80, 20]},
    )

    assert "baseline_metrics" in results
    assert set(results["perturbations"].keys()) == {"gaussian_noise", "compression"}
    for pert_name, curve in results["perturbations"].items():
        assert len(curve) == 2
        for point in curve:
            assert "severity" in point
            assert "accuracy" in point["metrics"]
    assert results["compression_in_training_mix"] is False

    plot_path = plot_robustness_curve(results, tmp_path / "figures" / "robustness_accuracy.png")
    assert plot_path.exists()
    assert plot_path.stat().st_size > 0
    assert plot_path.with_suffix(".csv").exists()
