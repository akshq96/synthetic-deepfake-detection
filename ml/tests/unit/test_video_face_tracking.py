from __future__ import annotations

import numpy as np
import pytest

from ml.data_pipeline.face_detector import FaceDetection, FaceDetector
from ml.video.face_tracking import iou, track_faces


class ScriptedFaceDetector(FaceDetector):
    """Test-only detector that returns a different, pre-scripted set of
    detections on each successive call — lets tracking tests simulate faces
    moving/appearing/disappearing across frames, which the pixel-content-
    blind MockFaceDetector can't do (it always returns the same detections).
    """

    def __init__(self, detections_per_call: list[list[FaceDetection]]):
        self._script = list(detections_per_call)
        self._call_index = 0

    def detect(self, image_bgr: np.ndarray) -> list[FaceDetection]:
        if self._call_index >= len(self._script):
            raise AssertionError("ScriptedFaceDetector called more times than scripted")
        result = self._script[self._call_index]
        self._call_index += 1
        return result


def _frame(size: int = 64) -> np.ndarray:
    return np.zeros((size, size, 3), dtype=np.uint8)


def test_iou_identical_boxes_is_one():
    box = (10, 10, 50, 50)
    assert iou(box, box) == pytest.approx(1.0)


def test_iou_disjoint_boxes_is_zero():
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_iou_partial_overlap():
    # box A: 0..10 x 0..10 (area 100). box B: 5..15 x 0..10 (area 100).
    # intersection: 5..10 x 0..10 (area 50). union = 100+100-50=150.
    score = iou((0, 0, 10, 10), (5, 0, 15, 10))
    assert score == pytest.approx(50 / 150)


def test_track_faces_same_position_stays_one_track():
    det = FaceDetection(x1=10, y1=10, x2=40, y2=40, confidence=0.9)
    detector = ScriptedFaceDetector([[det], [det], [det]])
    frames = [(0, _frame()), (1, _frame()), (2, _frame())]

    tracks = track_faces(frames, face_detector=detector, fps=10.0, crop_size=16)

    assert len(tracks) == 1
    assert len(tracks[0].detections) == 3
    assert [d.frame_index for d in tracks[0].detections] == [0, 1, 2]


def test_track_faces_two_disjoint_faces_become_two_tracks():
    det_a = FaceDetection(x1=0, y1=0, x2=20, y2=20, confidence=0.9)
    det_b = FaceDetection(x1=40, y1=40, x2=60, y2=60, confidence=0.9)
    detector = ScriptedFaceDetector([[det_a, det_b]])
    frames = [(0, _frame())]

    tracks = track_faces(frames, face_detector=detector, fps=10.0, crop_size=16)

    assert len(tracks) == 2


def test_track_faces_disappearing_face_starts_new_track_on_reappearance():
    det_early = FaceDetection(x1=10, y1=10, x2=30, y2=30, confidence=0.9)
    det_late = FaceDetection(x1=10, y1=10, x2=30, y2=30, confidence=0.9)  # same position, later
    # Frame 1 has no detections (face disappeared) -> frame-to-frame IoU
    # matching has nothing to match frame 2 against, so it becomes a new track.
    detector = ScriptedFaceDetector([[det_early], [], [det_late]])
    frames = [(0, _frame()), (1, _frame()), (2, _frame())]

    tracks = track_faces(frames, face_detector=detector, fps=10.0, crop_size=16)

    assert len(tracks) == 2
    assert sum(len(t.detections) for t in tracks) == 2


def test_track_faces_no_detections_returns_no_tracks():
    detector = ScriptedFaceDetector([[], []])
    frames = [(0, _frame()), (1, _frame())]
    tracks = track_faces(frames, face_detector=detector, fps=10.0, crop_size=16)
    assert tracks == []


def test_track_faces_crops_have_requested_size():
    det = FaceDetection(x1=10, y1=10, x2=40, y2=40, confidence=0.9)
    detector = ScriptedFaceDetector([[det]])
    frames = [(0, _frame())]
    tracks = track_faces(frames, face_detector=detector, fps=10.0, crop_size=32)
    assert tracks[0].detections[0].face_crop.shape == (32, 32, 3)
