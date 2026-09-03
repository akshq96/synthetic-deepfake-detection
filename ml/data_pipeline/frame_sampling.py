"""Frame index sampling and extraction for video sources.

Two distinct sampling goals live in this module (both configurable, kept
separate from any single hardcoded default):
  - *training-set construction*: a fixed, balanced number of frames per video
    (e.g. 32) so no single video dominates the training set.
  - *inference* (ml/video/, Phase 8): denser/adaptive sampling for speed vs
    coverage trade-offs — that module reuses `sample_frame_indices` with its
    own config rather than duplicating the sampling math.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np


def sample_frame_indices(
    total_frames: int, n_samples: int, *, strategy: str = "uniform"
) -> list[int]:
    """Return up to `n_samples` frame indices in [0, total_frames).

    If total_frames <= n_samples, returns all indices (every frame) — never
    raises for short clips, so a 5-frame video with n_samples=32 just yields 5.
    """
    if total_frames <= 0:
        return []
    n_samples = min(n_samples, total_frames)
    if strategy == "uniform":
        # np.linspace endpoints included; round + dedupe to get exactly n_samples
        # (or fewer, for tiny total_frames) evenly spaced integer indices.
        indices = np.linspace(0, total_frames - 1, num=n_samples)
        return sorted({int(round(i)) for i in indices})
    elif strategy == "every_nth":
        step = max(1, total_frames // n_samples)
        return list(range(0, total_frames, step))[:n_samples]
    else:
        raise ValueError(f"unknown sampling strategy: {strategy!r}, expected 'uniform' or 'every_nth'")


def extract_frames(video_path: str, indices: list[int]) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (frame_index, frame_bgr) for each requested index, in ascending
    order, seeking via cv2.VideoCapture. Indices beyond the video's actual
    frame count are silently skipped (a video's reported frame count from
    cv2.CAP_PROP_FRAME_COUNT can be inaccurate for some codecs).
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"could not open video: {video_path}")
    try:
        for idx in sorted(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                continue
            yield idx, frame
    finally:
        cap.release()


def video_frame_count(video_path: str) -> int:
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"could not open video: {video_path}")
    try:
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
