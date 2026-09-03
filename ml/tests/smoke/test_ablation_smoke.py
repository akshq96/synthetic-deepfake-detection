"""Ablation sweep runs end to end on fixture data and produces one result
per grid point, with distinct config values actually applied per run.
"""

from __future__ import annotations

from pathlib import Path

from ml.data_pipeline.manifest import save_manifest
from ml.data_pipeline.split import assign_splits
from ml.evaluation.ablation import run_ablation_sweep
from ml.tests.fixtures.synthetic_fixture import generate_fixture_dataset
from ml.training.config import load_config


def test_ablation_sweep_smoke(tmp_path: Path):
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
            "run_name=ablation_smoke",
            "training.debug_n_samples=30",
            "data.synthetic_techniques=[freq_perturb]",
        ],
    )

    results = run_ablation_sweep(cfg, grid={"data.synthetic_ratio": [0.0, 0.3]})

    assert len(results) == 2
    assert results[0]["point"] == {"data.synthetic_ratio": 0.0}
    assert results[1]["point"] == {"data.synthetic_ratio": 0.3}
    for r in results:
        assert Path(r["run_summary"]["checkpoint_path"]).exists()
        assert "accuracy" in r["eval_metrics"]


def test_ablation_grid_points_cartesian_product():
    from ml.evaluation.ablation import _grid_points

    points = _grid_points({"a": [1, 2], "b": [3, 4]})
    assert points == [
        {"a": 1, "b": 3},
        {"a": 1, "b": 4},
        {"a": 2, "b": 3},
        {"a": 2, "b": 4},
    ]


def test_ablation_grid_points_empty_grid_is_one_point():
    from ml.evaluation.ablation import _grid_points

    assert _grid_points({}) == [{}]
