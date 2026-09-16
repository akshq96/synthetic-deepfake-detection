"""Build a real-data manifest from the FaceForensics++ C32 face crops used by
DeepScan (Kaggle mirror, MTCNN crops, 5 frames per video).

Folder layout expected (DeepScan ml_server/ff_splits):
    {train,val,test}/real/<video>_f<k>.jpg
    {train,val,test}/fake/<Method>_<target>_<source>_f<k>.jpg   Method in Deepfakes, Face2Face, FaceSwap, NeuralTextures
    unseen/fake/<target>_f<k>.jpg                                FaceShifter

The existing folder split is ignored: identities are re-split by the project's
own leakage-safe splitter (ml.data_pipeline.split.assign_splits), grouping by
the target video id, which is shared by a real video and all its manipulated
versions. unseen/real duplicates test/real and is skipped.

    python -m ml.scripts.build_ffpp_manifest --root <ff_splits> \
        --out data/manifests/ffpp_c32_split.parquet --frames-per-video 2
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from ml.data_pipeline.leakage_check import check_no_leakage
from ml.data_pipeline.manifest import build_manifest, make_row, save_manifest
from ml.data_pipeline.split import assign_splits

FAKE_RE = re.compile(r"^(Deepfakes|Face2Face|FaceSwap|NeuralTextures)_(\d{3})_(\d{3})_f(\d+)\.jpg$")
REAL_RE = re.compile(r"^(\d{3})_f(\d+)\.jpg$")
SHIFTER_RE = re.compile(r"^(\d{3})_(\d{3})_f(\d+)\.jpg$")  # FaceShifter: <target>_<source>_f<k>


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames-per-video", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    root = Path(args.root)
    rows = []
    for sub in ("train", "val", "test"):
        for p in sorted((root / sub / "real").glob("*.jpg")):
            m = REAL_RE.match(p.name)
            if m and int(m.group(2)) < args.frames_per_video:
                vid = m.group(1)
                rows.append(make_row(source_dataset="ffpp_c32", source_video_id=f"real_{vid}", identity_id=vid,
                                     file_path=str(p.resolve()), label="real", frame_index=int(m.group(2)),
                                     compression="c32"))
        for p in sorted((root / sub / "fake").glob("*.jpg")):
            m = FAKE_RE.match(p.name)
            if m and int(m.group(4)) < args.frames_per_video:
                method, target, source = m.group(1), m.group(2), m.group(3)
                rows.append(make_row(source_dataset="ffpp_c32", source_video_id=f"{method}_{target}_{source}",
                                     identity_id=target, file_path=str(p.resolve()), label="fake",
                                     manipulation_type=method, frame_index=int(m.group(4)), compression="c32"))
    for p in sorted((root / "unseen" / "fake").glob("*.jpg")):
        m = SHIFTER_RE.match(p.name)
        if m and int(m.group(3)) < args.frames_per_video:
            target, source = m.group(1), m.group(2)
            rows.append(make_row(source_dataset="ffpp_c32", source_video_id=f"FaceShifter_{target}_{source}",
                                 identity_id=target, file_path=str(p.resolve()), label="fake",
                                 manipulation_type="FaceShifter", frame_index=int(m.group(3)), compression="c32"))

    df = assign_splits(build_manifest(rows), ratios=(0.7, 0.15, 0.15), seed=args.seed)
    check_no_leakage(df)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_manifest(df, args.out)
    summary = df.groupby(["split", "label", "manipulation_type"]).size()
    print(summary.to_string())
    print(f"rows={len(df)} identities={df['identity_id'].nunique()} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
