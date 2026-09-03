from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ml.data_pipeline.leakage_check import check_no_leakage
from ml.data_pipeline.schema import SPLIT_TRAIN, SPLIT_VAL
from ml.synthetic.mixer import (
    build_technique,
    generate_synthetic_samples,
    mix_into_manifest,
    n_synthetic_for_ratio,
)


def test_n_synthetic_for_ratio_zero():
    assert n_synthetic_for_ratio(100, 0.0) == 0


def test_n_synthetic_for_ratio_matches_target_proportion():
    n_real = 100
    ratio = 0.3
    n_synth = n_synthetic_for_ratio(n_real, ratio)
    achieved_ratio = n_synth / (n_real + n_synth)
    assert achieved_ratio == pytest.approx(ratio, abs=0.01)


def test_n_synthetic_for_ratio_rejects_out_of_range():
    with pytest.raises(ValueError, match="synthetic_ratio must be in"):
        n_synthetic_for_ratio(100, 1.0)
    with pytest.raises(ValueError, match="synthetic_ratio must be in"):
        n_synthetic_for_ratio(100, -0.1)


def test_build_technique_unknown_raises():
    with pytest.raises(ValueError, match="unknown synthetic technique"):
        build_technique("not_a_real_technique")


def test_generate_synthetic_samples_only_from_train_split(
    tmp_path: Path, fixture_manifest_split: pd.DataFrame
):
    synthetic = generate_synthetic_samples(
        fixture_manifest_split,
        techniques=["freq_perturb", "compression_artifact"],
        ratio=0.5,
        output_dir=tmp_path / "synthetic",
        seed=42,
    )
    assert len(synthetic) > 0
    assert (synthetic["split"] == SPLIT_TRAIN).all()
    assert synthetic["is_synthetic"].all()
    assert set(synthetic["synthetic_technique"].unique()) <= {"freq_perturb", "compression_artifact"}
    for p in synthetic["file_path"]:
        assert Path(p).exists()


def test_generate_synthetic_samples_inherits_identity_and_video(
    tmp_path: Path, fixture_manifest_split: pd.DataFrame
):
    synthetic = generate_synthetic_samples(
        fixture_manifest_split,
        techniques=["freq_perturb"],
        ratio=0.3,
        output_dir=tmp_path / "synthetic",
        seed=1,
    )
    train_real = fixture_manifest_split[fixture_manifest_split["split"] == SPLIT_TRAIN]
    valid_identities = set(train_real["identity_id"])
    valid_videos = set(train_real["source_video_id"])
    assert set(synthetic["identity_id"]).issubset(valid_identities)
    assert set(synthetic["source_video_id"]).issubset(valid_videos)


def test_mix_into_manifest_ratio_zero_is_noop(fixture_manifest_split: pd.DataFrame):
    mixed = mix_into_manifest(
        fixture_manifest_split, ratio=0.0, techniques=[], output_dir="/tmp/unused", seed=1
    )
    assert mixed is fixture_manifest_split


def test_mix_into_manifest_preserves_leakage_safety(tmp_path: Path, fixture_manifest_split: pd.DataFrame):
    mixed = mix_into_manifest(
        fixture_manifest_split,
        ratio=0.4,
        techniques=["freq_perturb", "color_perturb"],
        output_dir=tmp_path / "synthetic",
        seed=7,
    )
    check_no_leakage(mixed)  # must not raise
    assert len(mixed) > len(fixture_manifest_split)


def test_mix_into_manifest_does_not_touch_val_or_test(tmp_path: Path, fixture_manifest_split: pd.DataFrame):
    mixed = mix_into_manifest(
        fixture_manifest_split,
        ratio=0.5,
        techniques=["freq_perturb"],
        output_dir=tmp_path / "synthetic",
        seed=3,
    )
    original_val = fixture_manifest_split[fixture_manifest_split["split"] == SPLIT_VAL]
    mixed_val = mixed[mixed["split"] == SPLIT_VAL]
    assert len(mixed_val) == len(original_val)
    assert set(mixed_val["sample_id"]) == set(original_val["sample_id"])


def test_mix_into_manifest_is_deterministic(tmp_path: Path, fixture_manifest_split: pd.DataFrame):
    mixed1 = mix_into_manifest(
        fixture_manifest_split,
        ratio=0.3,
        techniques=["freq_perturb"],
        output_dir=tmp_path / "run1",
        seed=99,
    )
    mixed2 = mix_into_manifest(
        fixture_manifest_split,
        ratio=0.3,
        techniques=["freq_perturb"],
        output_dir=tmp_path / "run2",
        seed=99,
    )
    assert len(mixed1) == len(mixed2)
