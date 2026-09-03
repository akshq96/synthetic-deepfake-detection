"""Manifest schema: the single source of truth for every sample in the pipeline.

One row per sample (an image, or a face-cropped video frame). The manifest is the
backbone of leakage prevention (via `identity_id` / `source_video_id`) and of the
synthetic-data ratio controls used throughout training and evaluation.
"""

from __future__ import annotations

import pandas as pd

LABEL_REAL = "real"
LABEL_FAKE = "fake"
VALID_LABELS = (LABEL_REAL, LABEL_FAKE)

SPLIT_TRAIN = "train"
SPLIT_VAL = "val"
SPLIT_TEST = "test"
VALID_SPLITS = (SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST)

# column -> pandas dtype. Nullable dtypes used where a value may legitimately be
# absent (e.g. frame_index for image-only datasets, face_bbox before cropping).
MANIFEST_DTYPES: dict[str, str] = {
    "sample_id": "string",
    "source_dataset": "string",
    "source_video_id": "string",
    "identity_id": "string",
    "frame_index": "Int64",  # nullable int
    "file_path": "string",
    "label": "string",
    "manipulation_type": "string",
    "is_synthetic": "boolean",
    "synthetic_technique": "string",
    "split": "string",
    "compression": "string",
    "face_bbox_x1": "Float64",
    "face_bbox_y1": "Float64",
    "face_bbox_x2": "Float64",
    "face_bbox_y2": "Float64",
    "quality_ok": "boolean",
}

MANIFEST_COLUMNS: list[str] = list(MANIFEST_DTYPES.keys())

# Columns that must never be null in a finalized (post-split) manifest.
REQUIRED_NON_NULL = (
    "sample_id",
    "source_dataset",
    "source_video_id",
    "identity_id",
    "file_path",
    "label",
    "is_synthetic",
)


class ManifestValidationError(ValueError):
    """Raised when a manifest DataFrame violates the schema contract."""


def empty_manifest() -> pd.DataFrame:
    """Return an empty, correctly-typed manifest DataFrame."""
    return pd.DataFrame({col: pd.Series(dtype=dtype) for col, dtype in MANIFEST_DTYPES.items()})


def validate_manifest(df: pd.DataFrame, *, require_split: bool = False) -> None:
    """Validate a manifest DataFrame against the schema.

    Raises ManifestValidationError with a specific, actionable message on failure.
    Does not mutate `df`.
    """
    missing_cols = [c for c in MANIFEST_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ManifestValidationError(f"manifest missing required columns: {missing_cols}")

    for col in REQUIRED_NON_NULL:
        if df[col].isna().any():
            bad = df[df[col].isna()]
            raise ManifestValidationError(
                f"column '{col}' has {len(bad)} null value(s); "
                f"example sample_id(s): {bad['sample_id'].head(3).tolist()}"
            )

    bad_labels = set(df["label"].dropna().unique()) - set(VALID_LABELS)
    if bad_labels:
        raise ManifestValidationError(f"invalid label value(s) found: {bad_labels}, expected one of {VALID_LABELS}")

    dup_ids = df["sample_id"][df["sample_id"].duplicated()]
    if len(dup_ids) > 0:
        raise ManifestValidationError(f"duplicate sample_id(s) found: {dup_ids.unique().tolist()[:5]}")

    if require_split:
        if df["split"].isna().any():
            raise ManifestValidationError("split column has null values but require_split=True")
        bad_splits = set(df["split"].dropna().unique()) - set(VALID_SPLITS)
        if bad_splits:
            raise ManifestValidationError(f"invalid split value(s): {bad_splits}, expected one of {VALID_SPLITS}")


def cast_manifest(df: pd.DataFrame) -> pd.DataFrame:
    """Cast a manifest DataFrame's columns to the canonical dtypes.

    Missing columns are added as all-null. Column order is normalized.
    """
    df = df.copy()
    for col, dtype in MANIFEST_DTYPES.items():
        if col not in df.columns:
            df[col] = pd.Series([pd.NA] * len(df), dtype=dtype)
        else:
            df[col] = df[col].astype(dtype)
    return df[MANIFEST_COLUMNS]
