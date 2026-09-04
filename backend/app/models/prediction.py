"""A `Prediction` is one detect-image or detect-video request's result.
For video, per-frame detail lives in `SuspiciousFrame` rows (only the top-K
most suspicious frames are persisted, not every sampled frame — the rest is
transient, matching ml.video.aggregate's top_suspicious_frames output)."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.db.base import Base
from backend.app.models.experiment import _uuid, _utcnow

INPUT_TYPES = ("image", "video")
PREDICTION_LABELS = ("real", "fake", "abstain")


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    input_type: Mapped[str] = mapped_column(String(8))
    file_path: Mapped[str] = mapped_column(String(1024))
    model_name: Mapped[str] = mapped_column(String(64))
    label: Mapped[str] = mapped_column(String(16))
    confidence: Mapped[float] = mapped_column(Float)
    fake_probability: Mapped[float] = mapped_column(Float)
    abstained: Mapped[bool] = mapped_column(Boolean, default=False)
    heatmap_path: Mapped[str | None] = mapped_column(String(1024), default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    # Detect-page telemetry (Milestone B) — all nullable: only a subset applies
    # to any given input_type (e.g. duration_seconds is video-only), and older
    # rows predate these fields entirely. Never fabricated — each is only set
    # when the pipeline actually computed it (see ml/video/pipeline.py,
    # backend/app/services/ml_bridge.py).
    n_frames_total: Mapped[int | None] = mapped_column(Integer, default=None)
    n_frames_analyzed: Mapped[int | None] = mapped_column(Integer, default=None)
    video_width: Mapped[int | None] = mapped_column(Integer, default=None)
    video_height: Mapped[int | None] = mapped_column(Integer, default=None)
    duration_seconds: Mapped[float | None] = mapped_column(Float, default=None)
    processing_time_ms: Mapped[float | None] = mapped_column(Float, default=None)
    n_faces_detected: Mapped[int | None] = mapped_column(Integer, default=None)
    model_run_id: Mapped[str | None] = mapped_column(String(64), default=None)
    original_path: Mapped[str | None] = mapped_column(String(1024), default=None)
    heatmap_only_path: Mapped[str | None] = mapped_column(String(1024), default=None)

    suspicious_frames: Mapped[list["SuspiciousFrame"]] = relationship(  # noqa: F821
        "SuspiciousFrame", back_populates="prediction", cascade="all, delete-orphan"
    )
    frame_scores: Mapped[list["FrameScore"]] = relationship(  # noqa: F821
        "FrameScore", back_populates="prediction", cascade="all, delete-orphan"
    )
    reports: Mapped[list["Report"]] = relationship(  # noqa: F821
        "Report", back_populates="prediction", cascade="all, delete-orphan"
    )
