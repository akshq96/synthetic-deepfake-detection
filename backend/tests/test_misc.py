from __future__ import annotations

from fastapi.testclient import TestClient


def test_health(client: TestClient):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_mlflow_runs_empty_when_no_experiments(client: TestClient):
    response = client.get("/api/mlflow/runs")
    assert response.status_code == 200
    assert response.json() == []


def test_models_compare_unknown_experiment_404(client: TestClient):
    response = client.get("/api/models/compare", params={"experiment_name": "does_not_exist"})
    assert response.status_code == 404


def test_static_files_mounted(client: TestClient):
    # No file exists yet, but the mount itself should respond 404 (not 500).
    response = client.get("/static/does_not_exist.png")
    assert response.status_code == 404
