"""End-to-end video analysis: extract frames -> detect/track faces -> predict
each tracked face-frame -> aggregate into a video-level verdict -> surface
suspicious frames with heatmap overlays. This is what the backend's video-
detect endpoint (Phase 9) calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ml.data_pipeline.face_detector import FaceDetector, MediapipeFaceDetector
from ml.datasets.image_dataset import default_transform
from ml.models.predictor import Predictor
from ml.training.calibrate import Calibration
from ml.video.aggregate import FramePrediction, aggregate_video, top_suspicious_frames
from ml.video.extract_frames import sample_inference_frames, video_fps
from ml.video.face_tracking import track_faces
from ml.xai.interface import Explainer
from ml.xai.overlay import save_overlay


@dataclass
class VideoAnalysisResult:
    video_label: str
    track_results: dict
    n_tracks: int
    n_frames_sampled: int
    suspicious_frames: list[dict]


def analyze_video(
    video_path: str,
    model: nn.Module,
    *,
    face_detector: FaceDetector | None = None,
    calibration: Calibration | None = None,
    explainer: Explainer | None = None,
    heatmap_output_dir: str | Path | None = None,
    image_size: int = 224,
    frame_interval: int = 15,
    max_frames: int = 64,
    crop_margin: float = 0.3,
    abstain_margin: float = 0.1,
    aggregation_method: str = "mean_prob",
    top_k_suspicious: int = 5,
    device: str | torch.device = "cpu",
) -> VideoAnalysisResult:
    face_detector = face_detector or MediapipeFaceDetector()
    fps = video_fps(video_path)

    frames = sample_inference_frames(video_path, frame_interval=frame_interval, max_frames=max_frames)
    tracks = track_faces(
        frames, face_detector=face_detector, fps=fps, crop_margin=crop_margin, crop_size=image_size
    )

    import cv2

    predictor = Predictor(model, calibration=calibration, abstain_margin=abstain_margin, device=device)
    transform = default_transform(image_size, train=False)

    frame_predictions: list[FramePrediction] = []
    crop_by_key: dict[tuple[int, int], np.ndarray] = {}
    for track in tracks:
        for tracked_face in track.detections:
            crop_by_key[(track.track_id, tracked_face.frame_index)] = tracked_face.face_crop
            image_rgb = cv2.cvtColor(tracked_face.face_crop, cv2.COLOR_BGR2RGB)
            image_tensor = transform(image_rgb).unsqueeze(0)
            prediction = predictor.predict_batch(image_tensor)[0]
            frame_predictions.append(
                FramePrediction(
                    track_id=track.track_id,
                    frame_index=tracked_face.frame_index,
                    timestamp=tracked_face.timestamp,
                    prediction=prediction,
                )
            )

    video_result = aggregate_video(frame_predictions, method=aggregation_method)
    suspicious = top_suspicious_frames(frame_predictions, k=top_k_suspicious)

    suspicious_out = []
    for fp in suspicious:
        entry = {
            "track_id": fp.track_id,
            "frame_index": fp.frame_index,
            "timestamp": fp.timestamp,
            "label": fp.prediction.label,
            "confidence": fp.prediction.confidence,
            "fake_probability": fp.prediction.fake_probability,
            "heatmap_path": None,
        }
        if explainer is not None and heatmap_output_dir is not None:
            crop = crop_by_key[(fp.track_id, fp.frame_index)]
            image_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            image_tensor = transform(image_rgb).unsqueeze(0)
            heatmap = explainer.explain(model, image_tensor, target_class=1)[0]
            out_path = Path(heatmap_output_dir) / f"track{fp.track_id}_frame{fp.frame_index}.png"
            save_overlay(crop, heatmap, out_path)
            entry["heatmap_path"] = str(out_path)
        suspicious_out.append(entry)

    return VideoAnalysisResult(
        video_label=video_result["video_label"],
        track_results=video_result["track_results"],
        n_tracks=video_result["n_tracks"],
        n_frames_sampled=len(frames),
        suspicious_frames=suspicious_out,
    )
