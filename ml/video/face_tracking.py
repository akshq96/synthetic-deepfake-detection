"""Per-frame face detection + a simple greedy IoU tracker to group detections
into per-person tracks across frames. A video may contain multiple faces or
no consistent single identity, so each track is scored independently later
(ml.video.aggregate) — a video-level verdict considers every track, not just
one assumed face.

Deliberately not a heavyweight tracker (no DeepSORT/motion model): frame-to-
frame greedy IoU matching is enough at this scope and keeps the module
dependency-free beyond what Phase 1's face detector already needs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ml.data_pipeline.face_crop import crop_and_resize
from ml.data_pipeline.face_detector import FaceDetection, FaceDetector


@dataclass
class TrackedFace:
    track_id: int
    frame_index: int
    timestamp: float
    bbox: tuple[float, float, float, float]
    confidence: float
    face_crop: np.ndarray


@dataclass
class FaceTrack:
    track_id: int
    detections: list[TrackedFace] = field(default_factory=list)


def iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def track_faces(
    frames: list[tuple[int, np.ndarray]],
    *,
    face_detector: FaceDetector,
    fps: float,
    crop_margin: float = 0.3,
    crop_size: int = 224,
    iou_threshold: float = 0.3,
) -> list[FaceTrack]:
    """`frames`: [(frame_index, frame_bgr), ...] in ascending order (e.g. from
    ml.video.extract_frames.sample_inference_frames). Frames with no detected
    face contribute nothing (no track entry) rather than a degenerate crop.
    """
    from ml.video.extract_frames import frame_index_to_timestamp

    tracks: dict[int, FaceTrack] = {}
    prev_frame_boxes: dict[int, tuple[float, float, float, float]] = {}  # track_id -> last bbox
    next_track_id = 0

    for frame_index, frame_bgr in frames:
        detections = face_detector.detect(frame_bgr)
        this_frame_boxes: dict[int, tuple[float, float, float, float]] = {}
        matched_track_ids: set[int] = set()

        for det in detections:
            best_track_id, best_iou = None, iou_threshold
            for track_id, prev_box in prev_frame_boxes.items():
                if track_id in matched_track_ids:
                    continue
                score = iou(det.as_tuple(), prev_box)
                if score >= best_iou:
                    best_track_id, best_iou = track_id, score

            if best_track_id is None:
                best_track_id = next_track_id
                next_track_id += 1
                tracks[best_track_id] = FaceTrack(track_id=best_track_id)

            matched_track_ids.add(best_track_id)
            this_frame_boxes[best_track_id] = det.as_tuple()

            try:
                crop = crop_and_resize(frame_bgr, det, margin=crop_margin, output_size=crop_size)
            except ValueError:
                continue  # degenerate bbox (see ml.data_pipeline.face_crop) — skip this detection

            tracks[best_track_id].detections.append(
                TrackedFace(
                    track_id=best_track_id,
                    frame_index=frame_index,
                    timestamp=frame_index_to_timestamp(frame_index, fps),
                    bbox=det.as_tuple(),
                    confidence=det.confidence,
                    face_crop=crop,
                )
            )

        prev_frame_boxes = this_frame_boxes

    return [t for t in tracks.values() if t.detections]
