from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ml.data_pipeline.manifest import (
    build_manifest,
    load_manifest,
    make_row,
    merge_manifests,
    save_manifest,
)


def _sample_rows():
    return [
        make_row(
            source_dataset="fixture",
            source_video_id="v1",
            identity_id="id1",
            file_path="a.jpg",
            label="real",
        ),
        make_row(
            source_dataset="fixture",
            source_video_id="v2",
            identity_id="id2",
            file_path="b.jpg",
            label="fake",
            manipulation_type="deepfakes",
        ),
    ]


def test_build_manifest_roundtrip_types():
    df = build_manifest(_sample_rows())
    assert len(df) == 2
    assert set(df["label"]) == {"real", "fake"}
    assert df["split"].isna().all()


def test_build_manifest_empty():
    df = build_manifest([])
    assert len(df) == 0


def test_save_and_load_manifest(tmp_path: Path):
    df = build_manifest(_sample_rows())
    path = tmp_path / "manifests" / "fixture.parquet"
    save_manifest(df, path)
    assert path.exists()
    loaded = load_manifest(path)
    assert len(loaded) == len(df)
    assert sorted(loaded["sample_id"]) == sorted(df["sample_id"])


def test_load_manifest_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="manifest not found"):
        load_manifest(tmp_path / "does_not_exist.parquet")


def test_merge_manifests_concatenates():
    df1 = build_manifest(_sample_rows())
    df2 = build_manifest(
        [
            make_row(
                source_dataset="fixture",
                source_video_id="v3",
                identity_id="id3",
                file_path="c.jpg",
                label="real",
                is_synthetic=True,
                synthetic_technique="blend_warp",
            )
        ]
    )
    merged = merge_manifests([df1, df2])
    assert len(merged) == 3
    assert merged["is_synthetic"].sum() == 1


def test_merge_manifests_rejects_duplicate_sample_id():
    df1 = build_manifest(_sample_rows())
    dup_row = df1.iloc[[0]].copy()
    df2 = pd.DataFrame(dup_row)
    with pytest.raises(ValueError, match="duplicate sample_id"):
        merge_manifests([df1, df2])
