"""SQLAlchemy engine/session setup, shared by the app and Alembic."""

from __future__ import annotations

from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from backend.app.core.config import settings


class Base(DeclarativeBase):
    pass


def _make_engine():
    connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
    return create_engine(settings.database_url, connect_args=connect_args)


engine = _make_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_all_tables() -> None:
    """Dev/test convenience — creates tables directly from the ORM metadata
    without Alembic. Real deployments should use `alembic upgrade head`
    instead (see backend/alembic/); this stays handy for the SQLite-backed
    test suite and a from-scratch local run.
    """
    from backend.app import models  # noqa: F401 — ensures every model is registered on Base.metadata

    Base.metadata.create_all(bind=engine)
