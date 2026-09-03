from __future__ import annotations

import numpy as np
import pytest

from ml.data_pipeline.face_crop import crop_and_resize, expand_bbox
from ml.data_pipeline.face_detector import FaceDetection


def test_expand_bbox_grows_by_margin_and_clips():
    det = FaceDetection(x1=40, y1=40, x2=60, y2=60, confidence=0.9)
    x1, y1, x2, y2 = expand_bbox(det, margin=0.5, image_w=100, image_h=100)
    # width/height 20 -> margin 0.5 -> expand by 10 each side
    assert (x1, y1, x2, y2) == (30, 30, 70, 70)


def test_expand_bbox_clips_to_image_bounds():
    det = FaceDetection(x1=5, y1=5, x2=15, y2=15, confidence=0.9)
    x1, y1, x2, y2 = expand_bbox(det, margin=2.0, image_w=20, image_h=20)
    assert x1 == 0 and y1 == 0
    assert x2 <= 20 and y2 <= 20


def test_crop_and_resize_output_shape():
    img = np.random.default_rng(0).integers(0, 256, size=(100, 100, 3), dtype=np.uint8)
    det = FaceDetection(x1=20, y1=20, x2=60, y2=60, confidence=0.95)
    out = crop_and_resize(img, det, margin=0.2, output_size=32)
    assert out.shape == (32, 32, 3)
    assert out.dtype == np.uint8


def test_crop_and_resize_degenerate_bbox_raises():
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    det = FaceDetection(x1=49, y1=49, x2=49.4, y2=49.4, confidence=0.5)
    with pytest.raises(ValueError, match="degenerate crop region"):
        crop_and_resize(img, det, margin=0.0, output_size=32)
