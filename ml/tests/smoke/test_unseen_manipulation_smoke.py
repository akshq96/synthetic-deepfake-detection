"""Phase 5 smoke test: leave-one-manipulation-out training+eval runs end to
end and produces per-type results (including graceful skip when a held-out
type has no eval-split samples at fixture scale) — the mechanism the primary
research question is answered with, once run on real data. Definition of
done for Phase 5's unseen-manipulation half.
"""

from __future__ import annotations

from pathlib import Path

from ml.data_pipeline.manifest import save_manifest
from ml.data_pipeline.split import assign_splits
from ml.evaluation.experiments.unseen_manipulation import run_unseen_manipulation_experiment
from ml.tests.fixtures.synthetic_fixture import generate_fixture_dataset
from ml.training.config import load_config


def test_unseen_manipulation_smoke(tmp_path: Path):
    raw = generate_fixture_dataset(
        tmp_path / "raw", n_identities_per_label=4, fake_manipulation_types=("typeA", "typeB")
    )
    split = assign_splits(raw, seed=42)
    manifest_path = tmp_path / "manifests" / "fixture_split.parquet"
    save_manifest(split, manifest_path)

    artifacts_root = tmp_path / "artifacts"
    cfg = load_config(
        Path("ml/configs/cnn_baseline_debug.yaml"),
        cli_overrides=[
            f"data.manifest_path={manifest_path}",
            f"artifacts_root={artifacts_root}",
            "run_name=unseen_manip_smoke",
            "training.debug_n_samples=40",  # generous enough to keep both types represented
        ],
    )

    results = run_unseen_manipulation_experiment(cfg, held_out_types=["typeA", "typeB"])

    assert set(results.keys()) == {"typeA", "typeB"}
    # typeA has val-split samples at this fixture/seed -> should actually evaluate.
    assert results["typeA"]["skipped"] is False
    assert "eval_metrics" in results["typeA"]
    assert results["typeA"]["n_eval_samples"] > 0
    assert Path(results["typeA"]["train_summary"]["checkpoint_path"]).exists()

    # typeB has zero val-split samples at this fixture/seed -> must skip
    # gracefully rather than raise.
    assert results["typeB"]["skipped"] is True
    assert "reason" in results["typeB"]


def test_list_fake_manipulation_types(tmp_path: Path):
    from ml.evaluation.experiments.unseen_manipulation import list_fake_manipulation_types

    raw = generate_fixture_dataset(
        tmp_path / "raw", n_identities_per_label=4, fake_manipulation_types=("typeA", "typeB")
    )
    types = list_fake_manipulation_types(raw)
    assert types == ["typeA", "typeB"]
