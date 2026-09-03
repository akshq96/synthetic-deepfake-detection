"""Leakage-safe train/val/test split assignment.

Splits are assigned at the *group* level (identity_id, falling back to
source_video_id when identity is unknown) using a deterministic hash of the
group key + seed. This avoids depending on any RNG library's internal state
(unlike sklearn's GroupShuffleSplit) so a given (manifest, seed, ratios) always
produces byte-identical split assignments, which matters for reproducibility.

Assignment is done independently within each `source_dataset` and, within that,
lightly stratified by each group's majority label, so real/fake balance is
approximately preserved in each split without ever splitting a single identity
(or a single video's frames) across two splits.
"""

from __future__ import annotations

import hashlib

import pandas as pd

from ml.data_pipeline.schema import SPLIT_TEST, SPLIT_TRAIN, SPLIT_VAL, validate_manifest


def _group_key_column(df: pd.DataFrame) -> pd.Series:
    """The grouping key used for split assignment: identity_id, or
    source_video_id where identity_id is missing/empty (documented per-dataset
    in docs/methodology.md once written — not every dataset labels identities).
    """
    identity = df["identity_id"].astype("string")
    fallback = df["source_video_id"].astype("string")
    key = identity.where(identity.notna() & (identity.str.len() > 0), fallback)
    return key


def _hash_fraction(key: str, seed: int) -> float:
    """Deterministic, uniform-ish value in [0, 1) for a given key+seed."""
    h = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).hexdigest()
    # Use the first 15 hex digits (60 bits) as the numerator — plenty of resolution.
    return int(h[:15], 16) / float(16**15)


def assign_splits(
    df: pd.DataFrame,
    *,
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42,
) -> pd.DataFrame:
    """Return a copy of `df` with the `split` column populated.

    Guarantees: every row sharing a group key (identity_id, or source_video_id
    fallback) within the same source_dataset receives the same split. Ratios
    apply per (source_dataset, majority label of the group) stratum.
    """
    validate_manifest(df, require_split=False)
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1.0, got {ratios} (sum={sum(ratios)})")
    train_r, val_r, _test_r = ratios

    out = df.copy()
    out["_group_key"] = _group_key_column(out)

    # Majority label per (source_dataset, group_key) — used only to stratify
    # split assignment, not written back to the manifest.
    majority_label = (
        out.groupby(["source_dataset", "_group_key"])["label"]
        .agg(lambda s: s.value_counts().idxmax())
        .rename("_majority_label")
    )
    out = out.join(majority_label, on=["source_dataset", "_group_key"])

    def _assign(row_key: tuple[str, str]) -> str:
        dataset, group_key = row_key
        frac = _hash_fraction(f"{dataset}:{group_key}", seed)
        if frac < train_r:
            return SPLIT_TRAIN
        elif frac < train_r + val_r:
            return SPLIT_VAL
        else:
            return SPLIT_TEST

    strata_keys = out[["source_dataset", "_majority_label", "_group_key"]].drop_duplicates()
    # Salt the hash with the stratum label too, so real/fake groups don't
    # correlate in which fractional bucket they land in.
    strata_keys["split"] = strata_keys.apply(
        lambda r: _assign((f"{r['source_dataset']}|{r['_majority_label']}", r["_group_key"])),
        axis=1,
    )

    out = out.drop(columns=["split"]).merge(
        strata_keys[["source_dataset", "_group_key", "split"]],
        on=["source_dataset", "_group_key"],
        how="left",
    )
    out = out.drop(columns=["_group_key", "_majority_label"])
    out["split"] = out["split"].astype("string")
    validate_manifest(out, require_split=True)
    return out
