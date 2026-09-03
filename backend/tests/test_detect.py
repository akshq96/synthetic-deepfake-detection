from __future__ import annotations

import io

import cv2
import numpy as np
from fastapi.testclient import TestClient

from ml.tests.fixtures.synthetic_fixture import generate_fixture_video


def _tiny_jpeg_bytes(size: int = 64) -> bytes:
    img = np.random.default_rng(0).integers(0, 256, size=(size, size, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", img)
    assert ok
    return encoded.tobytes()


def test_detect_image_returns_prediction(client: TestClient):
    files = {"file": ("test.jpg", io.BytesIO(_tiny_jpeg_bytes()), "image/jpeg")}
    response = client.post("/api/detect/image", files=files)
    assert response.status_code == 200
    body = response.json()
    assert body["input_type"] == "image"
    assert body["label"] in ("real", "fake", "abstain")
    assert 0.0 <= body["confidence"] <= 1.0
    assert 0.0 <= body["fake_probability"] <= 1.0
    assert body["heatmap_path"] is not None
    assert "id" in body


def test_detect_image_rejects_bad_extension(client: TestClient):
    files = {"file": ("test.txt", io.BytesIO(b"not an image"), "text/plain")}
    response = client.post("/api/detect/image", files=files)
    assert response.status_code == 400


def test_detect_image_no_checkpoint_configured_returns_503(client: TestClient, monkeypatch):
    from backend.app.core.config import settings
    from backend.app.services import ml_bridge

    monkeypatch.setattr(settings, "default_checkpoint_path", None)
    ml_bridge._load_model_cached.cache_clear()

    files = {"file": ("test.jpg", io.BytesIO(_tiny_jpeg_bytes()), "image/jpeg")}
    response = client.post("/api/detect/image", files=files)
    assert response.status_code == 503


def test_get_prediction_roundtrip(client: TestClient):
    files = {"file": ("test.jpg", io.BytesIO(_tiny_jpeg_bytes()), "image/jpeg")}
    created = client.post("/api/detect/image", files=files).json()

    response = client.get(f"/api/detect/{created['id']}")
    assert response.status_code == 200
    assert response.json()["id"] == created["id"]


def test_get_prediction_not_found(client: TestClient):
    response = client.get("/api/detect/does-not-exist")
    assert response.status_code == 404


def test_detect_video_returns_prediction(client: TestClient, tmp_path):
    video_path = generate_fixture_video(tmp_path / "vid.avi", n_frames=10, size=64)
    with open(video_path, "rb") as f:
        files = {"file": ("test.avi", f, "video/x-msvideo")}
        response = client.post("/api/detect/video", files=files)
    assert response.status_code == 200
    body = response.json()
    assert body["input_type"] == "video"
    # A synthetic (non-photographic) fixture video has no detectable face
    # (MediapipeFaceDetector is the production default) -> no tracks -> the
    # pipeline abstains rather than crashing.
    assert body["label"] == "abstain"
    assert body["suspicious_frames"] == []


def test_detect_video_rejects_bad_extension(client: TestClient):
    files = {"file": ("test.txt", io.BytesIO(b"not a video"), "text/plain")}
    response = client.post("/api/detect/video", files=files)
    assert response.status_code == 400
