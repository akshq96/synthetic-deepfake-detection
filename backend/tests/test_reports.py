from __future__ import annotations

import io

import cv2
import numpy as np
from fastapi.testclient import TestClient


def _tiny_jpeg_bytes(size: int = 64) -> bytes:
    img = np.random.default_rng(0).integers(0, 256, size=(size, size, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", img)
    assert ok
    return encoded.tobytes()


def test_generate_and_export_report(client: TestClient):
    files = {"file": ("test.jpg", io.BytesIO(_tiny_jpeg_bytes()), "image/jpeg")}
    prediction = client.post("/api/detect/image", files=files).json()

    generate_response = client.post("/api/reports/generate", json={"prediction_id": prediction["id"]})
    assert generate_response.status_code == 200
    report = generate_response.json()
    assert report["prediction_id"] == prediction["id"]

    export_response = client.get(f"/api/reports/{report['id']}/export")
    assert export_response.status_code == 200
    assert export_response.headers["content-type"] == "application/pdf"
    assert len(export_response.content) > 0
    assert export_response.content.startswith(b"%PDF")


def test_list_reports(client: TestClient):
    files = {"file": ("test.jpg", io.BytesIO(_tiny_jpeg_bytes()), "image/jpeg")}
    prediction = client.post("/api/detect/image", files=files).json()
    generated = client.post("/api/reports/generate", json={"prediction_id": prediction["id"]}).json()

    response = client.get("/api/reports")
    assert response.status_code == 200
    reports = response.json()
    assert any(r["id"] == generated["id"] for r in reports)


def test_get_report(client: TestClient):
    files = {"file": ("test.jpg", io.BytesIO(_tiny_jpeg_bytes()), "image/jpeg")}
    prediction = client.post("/api/detect/image", files=files).json()
    generated = client.post("/api/reports/generate", json={"prediction_id": prediction["id"]}).json()

    response = client.get(f"/api/reports/{generated['id']}")
    assert response.status_code == 200
    assert response.json()["prediction_id"] == prediction["id"]


def test_get_report_unknown_id_404(client: TestClient):
    response = client.get("/api/reports/does-not-exist")
    assert response.status_code == 404


def test_generate_report_unknown_prediction_404(client: TestClient):
    response = client.post("/api/reports/generate", json={"prediction_id": "does-not-exist"})
    assert response.status_code == 404


def test_export_report_unknown_id_404(client: TestClient):
    response = client.get("/api/reports/does-not-exist/export")
    assert response.status_code == 404
