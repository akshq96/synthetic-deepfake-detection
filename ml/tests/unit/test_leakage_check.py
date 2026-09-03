from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ml.data_pipeline.leakage_check import LeakageError, check_no_leakage
from ml.data_pipeline.manifest import save_manifest


def test_check_no_leakage_passes_on_valid_split(fixture_manifest_split: pd.DataFrame):
    check_no_leakage(fixture_manifest_split)  # must not raise


def test_check_no_leakage_detects_identity_overlap(fixture_manifest_split: pd.DataFrame):
    df = fixture_manifest_split.copy()
    # Pick an identity with >1 row and force one of its rows into a different
    # split than the rest — this is exactly the leakage scenario the checker
    # exists to catch.
    identity_counts = df["identity_id"].value_counts()
    target_identity = identity_counts[identity_counts > 1].index[0]
    idx = df[df["identity_id"] == target_identity].index
    current_splits = df.loc[idx, "split"]
    other_split = next(s for s in ("train", "val", "test") if s not in set(current_splits))
    df.loc[idx[0], "split"] = other_split

    with pytest.raises(LeakageError, match="identity_id overlaps"):
        check_no_leakage(df)


def test_check_no_leakage_detects_video_overlap(fixture_manifest_split: pd.DataFrame):
    df = fixture_manifest_split.copy()
    video_counts = df["source_video_id"].value_counts()
    target_video = video_counts[video_counts > 1].index[0]
    idx = df[df["source_video_id"] == target_video].index
    current_splits = df.loc[idx, "split"]
    other_split = next(s for s in ("train", "val", "test") if s not in set(current_splits))
    # Also change identity_id for this one row so the identity-level check
    # doesn't fire first and mask the video-level check under test.
    df.loc[idx[0], "identity_id"] = df.loc[idx[0], "identity_id"] + "_isolated"
    df.loc[idx[0], "split"] = other_split

    with pytest.raises(LeakageError, match="source_video_id overlaps"):
        check_no_leakage(df)


def test_leakage_check_cli_passes(tmp_path: Path, fixture_manifest_split: pd.DataFrame, capsys):
    manifest_path = tmp_path / "manifest.parquet"
    save_manifest(fixture_manifest_split, manifest_path)
    # _main reads sys.argv via argparse; invoke it directly with a patched argv.
    import sys

    old_argv = sys.argv
    try:
        sys.argv = ["leakage_check", "--manifest", str(manifest_path)]
        from ml.data_pipeline.leakage_check import _main as main_fn

        code = main_fn()
    finally:
        sys.argv = old_argv
    assert code == 0
    assert "Leakage check passed" in capsys.readouterr().out


def test_leakage_check_cli_fails_on_bad_manifest(
    tmp_path: Path, fixture_manifest_split: pd.DataFrame, capsys
):
    df = fixture_manifest_split.copy()
    identity_counts = df["identity_id"].value_counts()
    target_identity = identity_counts[identity_counts > 1].index[0]
    idx = df[df["identity_id"] == target_identity].index
    other_split = next(
        s for s in ("train", "val", "test") if s not in set(df.loc[idx, "split"])
    )
    df.loc[idx[0], "split"] = other_split

    manifest_path = tmp_path / "bad_manifest.parquet"
    save_manifest(df, manifest_path)

    import sys

    old_argv = sys.argv
    try:
        sys.argv = ["leakage_check", "--manifest", str(manifest_path)]
        from ml.data_pipeline.leakage_check import _main as main_fn

        code = main_fn()
    finally:
        sys.argv = old_argv
    assert code == 1
    assert "LEAKAGE CHECK FAILED" in capsys.readouterr().err
