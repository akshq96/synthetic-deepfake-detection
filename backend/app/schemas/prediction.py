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
