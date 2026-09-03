"""Frame sampling for video *inference* — a different goal from the fixed
32-per-video sampling ml.data_pipeline.frame_sampling uses to build a
balanced training set. Inference wants speed/coverage at a fixed cadence
(every Nth frame, up to a cap) rather than a fixed count evenly spread over
the whole video, so it reuses the same low-level sample/extract functions
with its own defaults rather than duplicating the sampling math.
"""

from __future__ import annotations

import numpy as np

from ml.data_pipeline.frame_sampling import extract_frames, sample_frame_indices, video_frame_count


def sample_inference_frame_indices(
    total_frames: int, *, frame_interval: int = 15, max_frames: int = 64
) -> list[int]:
    if total_frames <= 0:
        return []
    indices = list(range(0, total_frames, frame_interval))
    return indices[:max_frames]


def sample_inference_frames(
    video_path: str, *, frame_interval: int = 15, max_frames: int = 64
) -> list[tuple[int, np.ndarray]]:
    """Returns [(frame_index, frame_bgr), ...] in ascending frame order."""
    total_frames = video_frame_count(video_path)
    indices = sample_inference_frame_indices(
        total_frames, frame_interval=frame_interval, max_frames=max_frames
    )
    return list(extract_frames(video_path, indices))


def frame_index_to_timestamp(frame_index: int, fps: float) -> float:
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    return frame_index / fps


def video_fps(video_path: str) -> float:
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"could not open video: {video_path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        return float(fps) if fps and fps > 0 else 25.0  # fall back to a documented default
    finally:
        cap.release()
