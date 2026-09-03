"""A `Run` tracks one launched training/experiment subprocess (see
backend/app/services/job_launcher.py). `status` values: pending, running,
completed, failed — kept as a plain string (not a DB-native enum type) so
SQLite and Postgres behave identically and adding a new status never needs a
migration, validated instead at the Pydantic schema layer.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.db.base import Base
from backend.app.models.experiment import _uuid, _utcnow

RUN_STATUSES = ("pending", "running", "completed", "failed")


class Run(Base):
    __tablename__ = "runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    experiment_id: Mapped[str] = mapped_column(ForeignKey("experiments.id"), index=True)
    run_name: Mapped[str] = mapped_column(String(255))
    mlflow_run_id: Mapped[str | None] = mapped_column(String(64), default=None)
    config_snapshot: Mapped[str] = mapped_column(Text)  # JSON-encoded resolved config
    status: Mapped[str] = mapped_column(String(16), default="pending", index=True)
    pid: Mapped[int | None] = mapped_column(Integer, default=None)
    error_message: Mapped[str | None] = mapped_column(Text, default=None)
    log_path: Mapped[str | None] = mapped_column(String(1024), default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)

    experiment: Mapped["Experiment"] = relationship("Experiment", back_populates="runs")  # noqa: F821
