from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.db.base import Base
from backend.app.models.experiment import _uuid, _utcnow


class Report(Base):
    __tablename__ = "reports"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    prediction_id: Mapped[str] = mapped_column(ForeignKey("predictions.id"), index=True)
    pdf_path: Mapped[str] = mapped_column(String(1024))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    prediction: Mapped["Prediction"] = relationship("Prediction", back_populates="reports")  # noqa: F821
