"""Face detection abstraction.

`FaceDetector` is an ABC so the production detector (MediaPipe) can be swapped
for an alternative (e.g. MTCNN via facenet-pytorch) later without touching any
calling code, and so unit tests can use a deterministic `MockFaceDetector`
instead of depending on a real CV model's accuracy on a fixture image.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FaceDetection:
    """A single detected face, bbox in absolute pixel coordinates (x1, y1, x2, y2)."""

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)


class FaceDetector(ABC):
    @abstractmethod
    def detect(self, image_bgr: np.ndarray) -> list[FaceDetection]:
        """Detect faces in a BGR image (as returned by cv2.imread). Returns a
        list of FaceDetection, sorted by confidence descending. Empty list if
        no face is found — callers must handle that (mark quality_ok=False),
        never assume at least one detection.
        """
        raise NotImplementedError


class MockFaceDetector(FaceDetector):
    """Deterministic test double: always returns the same configured bbox(es),
    regardless of input pixels. Used to unit-test crop-margin/resize logic
    without depending on a real model detecting a real face in a fixture image.
    """

    def __init__(self, detections: list[FaceDetection] | None = None):
        self._detections = detections if detections is not None else [
            FaceDetection(x1=10, y1=10, x2=90, y2=90, confidence=0.99)
        ]

    def detect(self, image_bgr: np.ndarray) -> list[FaceDetection]:
        return list(self._detections)


class MediapipeFaceDetector(FaceDetector):
    """Production detector using MediaPipe's Face Detection solution (BlazeFace).

    CPU-fast, pip-installable with no compiled toolchain — chosen over
    dlib/MTCNN for that reason (see docs/methodology.md once written). If
    MediaPipe's accuracy proves insufficient on a real dataset, swap in an
    alternative FaceDetector implementation without changing any caller.

    Requires mediapipe <1.0 (pinned in pyproject.toml): mediapipe 1.0 dropped
    the legacy `solutions` API in favor of a Tasks API whose face-detector
    graph was found to crash natively (native abort, not a Python exception)
    on macOS in this project's dev environment.
    """

    def __init__(self, min_confidence: float = 0.5, model_selection: int = 1):
        try:
            import mediapipe as mp
        except ImportError as e:  # pragma: no cover - exercised only when mediapipe missing
            raise ImportError(
                "mediapipe is required for MediapipeFaceDetector. Install the project's "
                "base dependencies: pip install -e ."
            ) from e
        self._mp_face_detection = mp.solutions.face_detection
        self._min_confidence = min_confidence
        self._model_selection = model_selection

    def detect(self, image_bgr: np.ndarray) -> list[FaceDetection]:
        import cv2

        h, w = image_bgr.shape[:2]
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        detections: list[FaceDetection] = []
        with self._mp_face_detection.FaceDetection(
            model_selection=self._model_selection,
            min_detection_confidence=self._min_confidence,
        ) as detector:
            result = detector.process(image_rgb)
            if result.detections:
                for det in result.detections:
                    box = det.location_data.relative_bounding_box
                    x1 = max(0.0, box.xmin * w)
                    y1 = max(0.0, box.ymin * h)
                    x2 = min(float(w), (box.xmin + box.width) * w)
                    y2 = min(float(h), (box.ymin + box.height) * h)
                    score = det.score[0] if det.score else 0.0
                    detections.append(FaceDetection(x1, y1, x2, y2, confidence=float(score)))
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections
