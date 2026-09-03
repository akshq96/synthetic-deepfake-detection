"""Generates a tiny, fully synthetic (not real faces) fixture dataset on the
fly for tests — deterministic, no binary files committed to the repo.

Produces small solid-color-plus-noise images (not real faces; real-face
detection is exercised separately in the MediaPipe integration test) grouped
into identities and source videos, so tests can exercise manifest building,
leakage-safe splitting, and the leakage checker against realistic group
structure (multiple frames per video, multiple videos per identity).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ml.data_pipeline.manifest import build_manifest, make_row
from ml.data_pipeline.schema import LABEL_FAKE, LABEL_REAL

N_IDENTITIES_PER_LABEL = 2
N_VIDEOS_PER_IDENTITY = 2
N_FRAMES_PER_VIDEO = 3
IMAGE_SIZE = 64


def _make_image(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base_color = rng.integers(0, 256, size=3)
    img = np.tile(base_color, (IMAGE_SIZE, IMAGE_SIZE, 1)).astype(np.uint8)
    noise = rng.integers(-20, 21, size=img.shape)
    img = np.clip(img.astype(int) + noise, 0, 255).astype(np.uint8)
    return img


def generate_fixture_dataset(
    root: Path,
    *,
    source_dataset: str = "fixture",
    n_identities_per_label: int = N_IDENTITIES_PER_LABEL,
    fake_manipulation_types: tuple[str, ...] = ("fixture_fake",),
) -> pd.DataFrame:
    """Write fixture images under `root/images/` and return their manifest
    (without splits assigned — call ml.data_pipeline.split.assign_splits on
    the result, as a real pipeline run would).

    `fake_manipulation_types`: when more than one is given, fake identities
    are cycled through them (identity N gets type N % len(types)) — used by
    the unseen-manipulation experiment tests to exercise leave-one-
    manipulation-out evaluation on data that actually has manipulation-type
    variety, which the default single-type fixture doesn't provide.
    """
    import cv2

    root = Path(root)
    images_dir = root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    seed_counter = 0
    for label in (LABEL_REAL, LABEL_FAKE):
        for identity_idx in range(n_identities_per_label):
            identity_id = f"{label}_identity_{identity_idx}"
            manipulation_type = (
                "none"
                if label == LABEL_REAL
                else fake_manipulation_types[identity_idx % len(fake_manipulation_types)]
            )
            for video_idx in range(N_VIDEOS_PER_IDENTITY):
                video_id = f"{identity_id}_video_{video_idx}"
                for frame_idx in range(N_FRAMES_PER_VIDEO):
                    seed_counter += 1
                    img = _make_image(seed_counter)
                    file_name = f"{video_id}_frame_{frame_idx}.jpg"
                    file_path = images_dir / file_name
                    cv2.imwrite(str(file_path), img)
                    rows.append(
                        make_row(
                            source_dataset=source_dataset,
                            source_video_id=video_id,
                            identity_id=identity_id,
                            file_path=str(file_path),
                            label=label,
                            manipulation_type=manipulation_type,
                            is_synthetic=False,
                            frame_index=frame_idx,
                            quality_ok=True,
                        )
                    )
    return build_manifest(rows)


def total_fixture_samples() -> int:
    return 2 * N_IDENTITIES_PER_LABEL * N_VIDEOS_PER_IDENTITY * N_FRAMES_PER_VIDEO


def generate_fixture_video(path: Path, *, n_frames: int = 20, size: int = 64, fps: float = 10.0) -> Path:
    """Writes a tiny synthetic (non-face) video for exercising ml/video/ —
    face detection on it must use a MockFaceDetector (see
    ml.data_pipeline.face_detector), same reasoning as the image fixtures.
    """
    import cv2

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (size, size))
    if not writer.isOpened():
        raise RuntimeError(f"OpenCV could not open a VideoWriter for {path}")
    for i in range(n_frames):
        writer.write(_make_image(seed=i))
    writer.release()
    return path
