"""Build, save, and load sample manifests (Parquet-backed)."""

from __future__ import annotations

import uuid
from pathlib import Path

import pandas as pd

from ml.data_pipeline.schema import MANIFEST_COLUMNS, cast_manifest, validate_manifest


def new_sample_id() -> str:
    return str(uuid.uuid4())


def make_row(
    *,
    source_dataset: str,
    source_video_id: str,
    identity_id: str,
    file_path: str,
    label: str,
    manipulation_type: str = "none",
    is_synthetic: bool = False,
    synthetic_technique: str | None = None,
    frame_index: int | None = None,
    compression: str | None = None,
    face_bbox: tuple[float, float, float, float] | None = None,
    quality_ok: bool = True,
    sample_id: str | None = None,
) -> dict:
    """Construct a single manifest row as a plain dict.

    `split` is intentionally left unset (NA) here — split assignment is a
    separate, auditable step performed by `ml.data_pipeline.split`.
    """
    x1, y1, x2, y2 = face_bbox if face_bbox is not None else (None, None, None, None)
    return {
        "sample_id": sample_id or new_sample_id(),
        "source_dataset": source_dataset,
        "source_video_id": source_video_id,
        "identity_id": identity_id,
        "frame_index": frame_index,
        "file_path": file_path,
        "label": label,
        "manipulation_type": manipulation_type,
        "is_synthetic": is_synthetic,
        "synthetic_technique": synthetic_technique,
        "split": None,
        "compression": compression,
        "face_bbox_x1": x1,
        "face_bbox_y1": y1,
        "face_bbox_x2": x2,
        "face_bbox_y2": y2,
        "quality_ok": quality_ok,
    }


def build_manifest(rows: list[dict]) -> pd.DataFrame:
    """Build a validated, correctly-typed manifest DataFrame from row dicts."""
    if not rows:
        from ml.data_pipeline.schema import empty_manifest

        return empty_manifest()
    df = pd.DataFrame(rows)
    df = cast_manifest(df)
    validate_manifest(df, require_split=False)
    return df


def save_manifest(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    validate_manifest(df, require_split=False)
    df.to_parquet(path, index=False)


def load_manifest(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"manifest not found at {path}. Build it first via ml.data_pipeline.manifest "
            f"or ml.scripts.download_datasets."
        )
    df = pd.read_parquet(path)
    df = cast_manifest(df)
    validate_manifest(df, require_split=False)
    return df


def merge_manifests(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate multiple manifests (e.g. real + synthetic) into one, re-validated."""
    dfs = [d for d in dfs if len(d) > 0]
    if not dfs:
        from ml.data_pipeline.schema import empty_manifest

        return empty_manifest()
    merged = pd.concat(dfs, ignore_index=True)
    merged = cast_manifest(merged)
    validate_manifest(merged, require_split=False)
    dup = merged["sample_id"][merged["sample_id"].duplicated()]
    if len(dup) > 0:
        raise ValueError(f"merge produced duplicate sample_id(s): {dup.unique().tolist()[:5]}")
    return merged
