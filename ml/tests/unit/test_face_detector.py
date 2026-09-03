from __future__ import annotations

import numpy as np
import pytest

from ml.data_pipeline.face_detector import FaceDetection, MockFaceDetector


def test_mock_face_detector_returns_configured_detection():
    det = FaceDetection(x1=1, y1=2, x2=3, y2=4, confidence=0.5)
    detector = MockFaceDetector(detections=[det])
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    result = detector.detect(image)
    assert result == [det]


def test_mock_face_detector_default_detection():
    detector = MockFaceDetector()
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    result = detector.detect(image)
    assert len(result) == 1
    assert result[0].confidence > 0


def test_mock_face_detector_ignores_pixel_content():
    detector = MockFaceDetector()
    image_a = np.zeros((100, 100, 3), dtype=np.uint8)
    image_b = np.full((100, 100, 3), 255, dtype=np.uint8)
    assert detector.detect(image_a) == detector.detect(image_b)


def test_mediapipe_face_detector_wiring_smoke():
    """Integration smoke test: verifies the MediaPipe API is wired up correctly
    (imports, model loads, process() call succeeds, output is well-formed) on a
    plain noise image. Does NOT assert a face is actually found — that
    requires a real photographic face image, which is validated separately
    once real datasets are available (Phase 1's later, data-dependent step),
    not via a committed fixture.
    """
    mediapipe = pytest.importorskip("mediapipe")
    from ml.data_pipeline.face_detector import MediapipeFaceDetector

    detector = MediapipeFaceDetector(min_confidence=0.5)
    image = np.random.default_rng(0).integers(0, 256, size=(200, 200, 3), dtype=np.uint8)
    result = detector.detect(image)
    assert isinstance(result, list)
    for det in result:
        assert det.x1 <= det.x2
        assert det.y1 <= det.y2
        assert 0.0 <= det.confidence <= 1.0
