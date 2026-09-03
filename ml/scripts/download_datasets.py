"""Validate local presence of raw datasets; print setup instructions if missing.

Does NOT download FaceForensics++, Celeb-DF v2, or DFDC automatically — all
three require accepting a dataset EULA / requesting access, which cannot be
scripted. This tool only checks whether the expected raw directory layout is
present under data/raw/ and, if not, prints where/how to request and place it.

Usage:
    python -m ml.scripts.download_datasets                 # check all
    python -m ml.scripts.download_datasets --dataset ffpp   # check one
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = REPO_ROOT / "data" / "raw"


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    display_name: str
    expected_subdirs: tuple[str, ...]
    instructions: str


DATASET_SPECS: dict[str, DatasetSpec] = {
    "ffpp": DatasetSpec(
        key="ffpp",
        display_name="FaceForensics++",
        expected_subdirs=("original_sequences", "manipulated_sequences"),
        instructions=(
            "1. Request access via the form linked from the official repo: "
            "https://github.com/ondyari/FaceForensics (Access section) — "
            "requires agreeing to the FaceForensics++ terms of use.\n"
            "  2. Once approved, use the provided download script to fetch the "
            "c23-compressed videos (recommended for a first pass; raw/c40 are "
            "also available).\n"
            "  3. Place the result under:\n"
            f"     {DATA_RAW / 'ffpp'}/original_sequences/...\n"
            f"     {DATA_RAW / 'ffpp'}/manipulated_sequences/<Deepfakes|Face2Face|"
            "FaceSwap|NeuralTextures>/...\n"
            "  This layout matches FaceForensics++'s own download-script output."
        ),
    ),
    "celebdf": DatasetSpec(
        key="celebdf",
        display_name="Celeb-DF v2",
        expected_subdirs=("Celeb-real", "Celeb-synthesis", "YouTube-real"),
        instructions=(
            "1. Request access per the instructions at: "
            "https://github.com/yuezunli/celeb-deepfakeforensics — the authors "
            "grant access after a request-form submission.\n"
            "  2. Place the result under:\n"
            f"     {DATA_RAW / 'celebdf'}/Celeb-real/...\n"
            f"     {DATA_RAW / 'celebdf'}/Celeb-synthesis/...\n"
            f"     {DATA_RAW / 'celebdf'}/YouTube-real/...\n"
        ),
    ),
    "dfdc": DatasetSpec(
        key="dfdc",
        display_name="DFDC (Deepfake Detection Challenge)",
        expected_subdirs=("train_sample_videos",),
        instructions=(
            "1. Easiest path: the DFDC sample/preview set is hosted on Kaggle — "
            "https://www.kaggle.com/c/deepfake-detection-challenge/data "
            "(free Kaggle account required; also directly usable from a Kaggle "
            "Notebook without downloading, which pairs well with this project's "
            "free-GPU training setup — see docs/kaggle_setup.md once written).\n"
            "  2. The full DFDC dataset (larger) is described at "
            "https://ai.meta.com/datasets/dfdc/ .\n"
            "  3. Place (or symlink, if working from a Kaggle Notebook) the "
            "result under:\n"
            f"     {DATA_RAW / 'dfdc'}/train_sample_videos/...\n"
            "     (each video's real/fake label and speaker id come from that "
            "directory's metadata.json)."
        ),
    ),
}


def check_dataset(key: str) -> bool:
    spec = DATASET_SPECS[key]
    root = DATA_RAW / key
    if not root.exists():
        print(f"[MISSING] {spec.display_name}: expected directory not found: {root}")
        print(f"  How to get it:\n  {spec.instructions}\n")
        return False

    missing_subdirs = [s for s in spec.expected_subdirs if not (root / s).exists()]
    if missing_subdirs:
        print(
            f"[INCOMPLETE] {spec.display_name}: {root} exists but is missing "
            f"expected subdirectories: {missing_subdirs}"
        )
        print(f"  How to get it:\n  {spec.instructions}\n")
        return False

    print(f"[OK] {spec.display_name}: found at {root}")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=list(DATASET_SPECS.keys()),
        default=None,
        help="Check a single dataset (default: check all)",
    )
    args = parser.parse_args(argv)

    keys = [args.dataset] if args.dataset else list(DATASET_SPECS.keys())
    results = {k: check_dataset(k) for k in keys}

    n_ok = sum(results.values())
    print(f"\n{n_ok}/{len(results)} dataset(s) ready.")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
