"""The full per-sampled-frame probability series for a video prediction —
every entry `ml/video/pipeline.py`'s `analyze_video()` computes, not just the
top-K subset that `SuspiciousFrame` persists. Powers the Detect page's
frame-level probability chart. Video predictions only; empty for images."""

from __future__ import annotations

from sqlalchemy import Float, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.db.base import Base
from backend.app.models.experiment import _uuid


class FrameScore(Base):
    __tablename__ = "frame_scores"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    prediction_id: Mapped[str] = mapped_column(ForeignKey("predictions.id"), index=True)
    track_id: Mapped[int] = mapped_column(Integer)
    frame_index: Mapped[int] = mapped_column(Integer)
    timestamp: Mapped[float] = mapped_column(Float)
    fake_probability: Mapped[float] = mapped_column(Float)

    prediction: Mapped["Prediction"] = relationship("Prediction", back_populates="frame_scores")  # noqa: F821
