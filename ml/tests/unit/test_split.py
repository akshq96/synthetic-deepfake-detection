from __future__ import annotations

import pandas as pd
import pytest

from ml.data_pipeline.split import assign_splits
from ml.data_pipeline.schema import VALID_SPLITS


def test_assign_splits_covers_all_rows(fixture_manifest_unsplit: pd.DataFrame):
    out = assign_splits(fixture_manifest_unsplit, seed=42)
    assert len(out) == len(fixture_manifest_unsplit)
    assert out["split"].isna().sum() == 0
    assert set(out["split"].unique()).issubset(set(VALID_SPLITS))


def test_assign_splits_is_deterministic(fixture_manifest_unsplit: pd.DataFrame):
    out1 = assign_splits(fixture_manifest_unsplit, seed=42)
    out2 = assign_splits(fixture_manifest_unsplit, seed=42)
    merged = out1[["sample_id", "split"]].merge(
        out2[["sample_id", "split"]], on="sample_id", suffixes=("_1", "_2")
    )
    assert (merged["split_1"] == merged["split_2"]).all()


def test_assign_splits_different_seed_can_differ(fixture_manifest_unsplit: pd.DataFrame):
    out1 = assign_splits(fixture_manifest_unsplit, seed=1)
    out2 = assign_splits(fixture_manifest_unsplit, seed=2)
    merged = out1[["sample_id", "split"]].merge(
        out2[["sample_id", "split"]], on="sample_id", suffixes=("_1", "_2")
    )
    # Not a hard guarantee for every possible dataset, but for this fixture's
    # group count, two different seeds should not produce an identical
    # assignment for every single row.
    assert not (merged["split_1"] == merged["split_2"]).all()


def test_assign_splits_never_splits_a_single_identity(fixture_manifest_unsplit: pd.DataFrame):
    out = assign_splits(fixture_manifest_unsplit, seed=42)
    splits_per_identity = out.groupby(["source_dataset", "identity_id"])["split"].nunique()
    assert (splits_per_identity == 1).all()


def test_assign_splits_never_splits_a_single_video(fixture_manifest_unsplit: pd.DataFrame):
    out = assign_splits(fixture_manifest_unsplit, seed=42)
    splits_per_video = out.groupby(["source_dataset", "source_video_id"])["split"].nunique()
    assert (splits_per_video == 1).all()


def test_assign_splits_rejects_bad_ratios(fixture_manifest_unsplit: pd.DataFrame):
    with pytest.raises(ValueError, match="ratios must sum to 1.0"):
        assign_splits(fixture_manifest_unsplit, ratios=(0.5, 0.5, 0.5), seed=42)
