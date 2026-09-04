from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class SuspiciousFrameOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    track_id: int
    frame_index: int
    timestamp: float
    fake_probability: float
    heatmap_path: str | None = None
    original_path: str | None = None
    heatmap_only_path: str | None = None


class FrameScoreOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    track_id: int
    frame_index: int
    timestamp: float
    fake_probability: float


class PredictionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    input_type: str
    model_name: str
    label: str
    confidence: float
    fake_probability: float
    abstained: bool
    heatmap_path: str | None = None
    created_at: datetime
    suspicious_frames: list[SuspiciousFrameOut] = []

    # Detect-page telemetry (Milestone B) — all nullable: only a subset
    # applies to any given input_type, and rows predating this feature have
    # none of it. Never fabricated — each is only set when the pipeline
    # actually computed it (see ml/video/pipeline.py, ml_bridge.py).
    n_frames_total: int | None = None
    n_frames_analyzed: int | None = None
    video_width: int | None = None
    video_height: int | None = None
    duration_seconds: float | None = None
    processing_time_ms: float | None = None
    n_faces_detected: int | None = None
    model_run_id: str | None = None
    original_path: str | None = None
    heatmap_only_path: str | None = None
    frame_scores: list[FrameScoreOut] = []
