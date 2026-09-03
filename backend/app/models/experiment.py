from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.db.base import Base


def _uuid() -> str:
    return str(uuid.uuid4())


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Experiment(Base):
    __tablename__ = "experiments"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    description: Mapped[str | None] = mapped_column(Text, default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    runs: Mapped[list["Run"]] = relationship(
        "Run", back_populates="experiment", cascade="all, delete-orphan"
    )
