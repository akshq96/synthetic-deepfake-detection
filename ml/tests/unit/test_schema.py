from __future__ import annotations

import pandas as pd
import pytest

from ml.data_pipeline.schema import (
    ManifestValidationError,
    cast_manifest,
    empty_manifest,
    validate_manifest,
)


def test_empty_manifest_has_correct_columns():
    df = empty_manifest()
    assert list(df.columns) == list(df.columns)
    assert len(df) == 0
    validate_manifest(df, require_split=False)


def test_validate_manifest_missing_column_raises():
    df = empty_manifest().drop(columns=["label"])
    with pytest.raises(ManifestValidationError, match="missing required columns"):
        validate_manifest(df)


def test_validate_manifest_rejects_null_required_field():
    df = pd.concat(
        [
            empty_manifest(),
            pd.DataFrame(
                [
                    {
                        "sample_id": "s1",
                        "source_dataset": "d",
                        "source_video_id": None,
                        "identity_id": "id1",
                        "file_path": "x.jpg",
                        "label": "real",
                        "is_synthetic": False,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    df = cast_manifest(df)
    with pytest.raises(ManifestValidationError, match="source_video_id"):
        validate_manifest(df)


def test_validate_manifest_rejects_invalid_label():
    df = cast_manifest(
        pd.DataFrame(
            [
                {
                    "sample_id": "s1",
                    "source_dataset": "d",
                    "source_video_id": "v1",
                    "identity_id": "id1",
                    "file_path": "x.jpg",
                    "label": "definitely_fake_but_not_a_valid_enum",
                    "is_synthetic": False,
                }
            ]
        )
    )
    with pytest.raises(ManifestValidationError, match="invalid label"):
        validate_manifest(df)


def test_validate_manifest_rejects_duplicate_sample_id():
    row = {
        "sample_id": "dup",
        "source_dataset": "d",
        "source_video_id": "v1",
        "identity_id": "id1",
        "file_path": "x.jpg",
        "label": "real",
        "is_synthetic": False,
    }
    df = cast_manifest(pd.DataFrame([row, row]))
    with pytest.raises(ManifestValidationError, match="duplicate sample_id"):
        validate_manifest(df)


def test_validate_manifest_require_split_enforced():
    row = {
        "sample_id": "s1",
        "source_dataset": "d",
        "source_video_id": "v1",
        "identity_id": "id1",
        "file_path": "x.jpg",
        "label": "real",
        "is_synthetic": False,
        "split": None,
    }
    df = cast_manifest(pd.DataFrame([row]))
    validate_manifest(df, require_split=False)  # ok without split
    with pytest.raises(ManifestValidationError, match="split column has null values"):
        validate_manifest(df, require_split=True)
