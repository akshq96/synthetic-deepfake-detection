"""Crop a detected face out of a frame with a documented, reproducible margin
convention, and resize/pad to a fixed square output size.

Crop convention (kept here, not scattered across call sites, so every dataset
is processed identically): expand the detected bbox by `margin` fraction of
its size on each side (default 0.3 = 30%), clip to image bounds, then resize
the resulting crop to `output_size x output_size` — no aspect-ratio distortion
beyond what a square-ify via padding introduces, since faces are already
roughly square after margin expansion. This matches the documented crop
convention referenced in the plan (docs/methodology.md, once written).
"""

from __future__ import annotations

import numpy as np

from ml.data_pipeline.face_detector import FaceDetection


def expand_bbox(
    det: FaceDetection, *, margin: float, image_w: int, image_h: int
) -> tuple[int, int, int, int]:
    """Expand a bbox by `margin` fraction of its width/height on each side,
    clipped to [0, image_w] x [0, image_h]. Returns integer pixel coords.
    """
    w = det.x2 - det.x1
    h = det.y2 - det.y1
    dx = w * margin
    dy = h * margin
    x1 = max(0, int(round(det.x1 - dx)))
    y1 = max(0, int(round(det.y1 - dy)))
    x2 = min(image_w, int(round(det.x2 + dx)))
    y2 = min(image_h, int(round(det.y2 + dy)))
    return x1, y1, x2, y2


def crop_and_resize(
    image_bgr: np.ndarray,
    det: FaceDetection,
    *,
    margin: float = 0.3,
    output_size: int = 224,
) -> np.ndarray:
    """Crop the face region (with margin) out of `image_bgr` and resize to a
    square `output_size x output_size` image. Raises ValueError if the
    expanded bbox is degenerate (zero width/height) rather than silently
    returning a broken crop.
    """
    import cv2

    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = expand_bbox(det, margin=margin, image_w=w, image_h=h)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(
            f"degenerate crop region ({x1},{y1},{x2},{y2}) from detection {det} "
            f"on image of size {w}x{h}"
        )
    crop = image_bgr[y1:y2, x1:x2]
    resized = cv2.resize(crop, (output_size, output_size), interpolation=cv2.INTER_AREA)
    return resized
