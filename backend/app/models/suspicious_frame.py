from __future__ import annotations

from sqlalchemy import Float, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.db.base import Base
from backend.app.models.experiment import _uuid


class SuspiciousFrame(Base):
    __tablename__ = "suspicious_frames"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    prediction_id: Mapped[str] = mapped_column(ForeignKey("predictions.id"), index=True)
    track_id: Mapped[int] = mapped_column(Integer)
    frame_index: Mapped[int] = mapped_column(Integer)
    timestamp: Mapped[float] = mapped_column(Float)
    fake_probability: Mapped[float] = mapped_column(Float)
    heatmap_path: Mapped[str | None] = mapped_column(String(1024), default=None)

    prediction: Mapped["Prediction"] = relationship("Prediction", back_populates="suspicious_frames")  # noqa: F821
