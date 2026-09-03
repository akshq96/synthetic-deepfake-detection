"""Leakage guard: no identity and no source video may appear in more than one split.

This is the single most safety-critical control in the pipeline — every scientific
claim downstream (baseline vs synthetic, unseen-manipulation generalization,
cross-dataset generalization, robustness) is invalid if train/val/test share an
identity or a source video. `check_no_leakage` is called both as a pytest and as a
CLI guard invoked by `ml.training.train` before any run starts; it hard-fails rather
than warning.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from ml.data_pipeline.manifest import load_manifest
from ml.data_pipeline.schema import validate_manifest


class LeakageError(ValueError):
    """Raised when the same identity or source video spans more than one split."""


def _find_cross_split_overlaps(df: pd.DataFrame, key_col: str) -> dict[str, set[str]]:
    """Return {key_value: set_of_splits} for every key that appears in >1 split,
    scoped within each source_dataset (the same identity_id string in two
    different datasets is not leakage — dataset ids are not globally unique).
    """
    overlaps: dict[str, set[str]] = {}
    keyed = df[["source_dataset", key_col, "split"]].dropna(subset=[key_col])
    grouped = keyed.groupby(["source_dataset", key_col])["split"].agg(lambda s: set(s))
    for (dataset, key_val), splits in grouped.items():
        if len(splits) > 1:
            overlaps[f"{dataset}:{key_val}"] = splits
    return overlaps


def check_no_leakage(df: pd.DataFrame) -> None:
    """Raise LeakageError if any identity_id or source_video_id crosses splits.

    Checks both keys independently: identity_id is the primary leakage vector,
    source_video_id is checked too in case identity labeling is missing/partial
    for a dataset (so distinct "unknown" identities don't mask a video-level leak).
    """
    validate_manifest(df, require_split=True)

    identity_overlaps = _find_cross_split_overlaps(df, "identity_id")
    video_overlaps = _find_cross_split_overlaps(df, "source_video_id")

    if identity_overlaps or video_overlaps:
        lines = ["Data leakage detected between splits:"]
        if identity_overlaps:
            lines.append(f"  identity_id overlaps ({len(identity_overlaps)}):")
            for key, splits in list(identity_overlaps.items())[:10]:
                lines.append(f"    {key} appears in splits {sorted(splits)}")
            if len(identity_overlaps) > 10:
                lines.append(f"    ... and {len(identity_overlaps) - 10} more")
        if video_overlaps:
            lines.append(f"  source_video_id overlaps ({len(video_overlaps)}):")
            for key, splits in list(video_overlaps.items())[:10]:
                lines.append(f"    {key} appears in splits {sorted(splits)}")
            if len(video_overlaps) > 10:
                lines.append(f"    ... and {len(video_overlaps) - 10} more")
        raise LeakageError("\n".join(lines))


def _main() -> int:
    parser = argparse.ArgumentParser(
        description="Hard-fail if any identity/source video spans more than one split."
    )
    parser.add_argument("--manifest", required=True, help="Path to a split manifest (.parquet)")
    args = parser.parse_args()

    df = load_manifest(args.manifest)
    try:
        check_no_leakage(df)
    except LeakageError as e:
        print(f"LEAKAGE CHECK FAILED for {args.manifest}\n\n{e}", file=sys.stderr)
        return 1
    print(f"Leakage check passed: {args.manifest} ({len(df)} samples, no cross-split overlap).")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
