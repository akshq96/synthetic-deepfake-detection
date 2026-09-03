from __future__ import annotations

from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ml.models.factory import ModelConfig, build_model
from ml.training.calibrate import Calibration
from ml.training.checkpoint import save_checkpoint


@pytest.fixture
def tiny_checkpoint(tmp_path: Path) -> dict:
    """A randomly-initialized (not trained) checkpoint — enough to exercise
    every code path (load, predict, explain) without needing real trained
    weights or a real dataset, per the project plan's integration-test design.
    """
    model = build_model(ModelConfig(name="efficientnetv2_s", pretrained=False))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    checkpoint_path = tmp_path / "checkpoint" / "latest.pt"
    save_checkpoint(checkpoint_path, model=model, optimizer=optimizer, epoch=0, global_step=1, best_metric=0.0)

    calibration_path = tmp_path / "checkpoint" / "temperature.json"
    Calibration.identity().save(calibration_path)

    return {
        "checkpoint_path": str(checkpoint_path),
        "calibration_path": str(calibration_path),
        "model_name": "efficientnetv2_s",
        "image_size": 64,
    }


@pytest.fixture
def client(tmp_path: Path, monkeypatch, tiny_checkpoint: dict) -> TestClient:
    from backend.app.core.config import settings

    monkeypatch.setattr(settings, "database_url", f"sqlite:///{tmp_path / 'test.db'}")
    monkeypatch.setattr(settings, "artifacts_root", tmp_path / "artifacts")
    monkeypatch.setattr(settings, "default_checkpoint_path", tiny_checkpoint["checkpoint_path"])
    monkeypatch.setattr(settings, "default_calibration_path", tiny_checkpoint["calibration_path"])
    monkeypatch.setattr(settings, "default_model_name", tiny_checkpoint["model_name"])
    monkeypatch.setattr(settings, "default_image_size", tiny_checkpoint["image_size"])

    # A fresh engine bound to the per-test SQLite file, independent of the
    # module-level engine backend.app.db.base created at import time from
    # the original settings.
    test_engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}", connect_args={"check_same_thread": False})
    TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)

    import backend.app.db.base as db_base

    monkeypatch.setattr(db_base, "engine", test_engine)
    monkeypatch.setattr(db_base, "SessionLocal", TestSessionLocal)

    from backend.app import models  # noqa: F401 — registers models on Base.metadata

    db_base.Base.metadata.create_all(bind=test_engine)

    from backend.app.services import ml_bridge

    ml_bridge._load_model_cached.cache_clear()  # avoid cross-test cache pollution (lru_cache is module-global)

    from backend.app.main import app

    def override_get_db():
        db = TestSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[db_base.get_db] = override_get_db

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()
