"""A `Prediction` is one detect-image or detect-video request's result.
For video, per-frame detail lives in `SuspiciousFrame` rows (only the top-K
most suspicious frames are persisted, not every sampled frame — the rest is
transient, matching ml.video.aggregate's top_suspicious_frames output)."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, String
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

    suspicious_frames: Mapped[list["SuspiciousFrame"]] = relationship(  # noqa: F821
        "SuspiciousFrame", back_populates="prediction", cascade="all, delete-orphan"
    )
    reports: Mapped[list["Report"]] = relationship(  # noqa: F821
        "Report", back_populates="prediction", cascade="all, delete-orphan"
    )
