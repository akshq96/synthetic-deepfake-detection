from __future__ import annotations

import time
from pathlib import Path

from fastapi.testclient import TestClient

from ml.data_pipeline.manifest import save_manifest
from ml.data_pipeline.split import assign_splits
from ml.tests.fixtures.synthetic_fixture import generate_fixture_dataset


def test_list_experiments_empty_initially(client: TestClient):
    response = client.get("/api/experiments")
    assert response.status_code == 200
    assert response.json() == []


def test_get_experiment_not_found(client: TestClient):
    response = client.get("/api/experiments/does-not-exist")
    assert response.status_code == 404


def test_synthetic_lab_run_launches_and_completes(client: TestClient, tmp_path: Path):
    raw = generate_fixture_dataset(tmp_path / "raw")
    split = assign_splits(raw, seed=42)
    manifest_path = tmp_path / "manifests" / "fixture_split.parquet"
    save_manifest(split, manifest_path)

    payload = {
        "experiment_name": "synthetic_lab_test",
        "run_name": "synthetic_lab_test_run",
        "base_config_path": "ml/configs/cnn_baseline_debug.yaml",
        "synthetic_ratio": 0.3,
        "synthetic_techniques": ["freq_perturb"],
        "manifest_path": str(manifest_path),
    }
    launch_response = client.post("/api/experiments/synthetic-lab/run", json=payload)
    assert launch_response.status_code == 200
    run = launch_response.json()
    assert run["status"] == "running"
    assert run["pid"] is not None

    # Poll to completion — a real subprocess (ml.training.train) is running;
    # debug-scale (1 epoch, a handful of samples, 64x64 CPU) should finish
    # in well under the timeout.
    deadline = time.time() + 90
    status = run["status"]
    while status == "running" and time.time() < deadline:
        time.sleep(1.0)
        status_response = client.get(f"/api/experiments/{run['id']}/status")
        assert status_response.status_code == 200
        status = status_response.json()["status"]

    assert status == "completed", f"run did not complete in time (last status={status})"

    experiment_response = client.get(f"/api/experiments/{run['experiment_id']}")
    assert experiment_response.status_code == 200
    experiment = experiment_response.json()
    assert experiment["name"] == "synthetic_lab_test"
    assert len(experiment["runs"]) == 1
    assert experiment["runs"][0]["mlflow_run_id"] is not None


def test_synthetic_lab_run_missing_config_returns_400(client: TestClient):
    payload = {
        "experiment_name": "e",
        "run_name": "r",
        "base_config_path": "ml/configs/does_not_exist.yaml",
    }
    response = client.post("/api/experiments/synthetic-lab/run", json=payload)
    assert response.status_code == 400


def test_run_status_not_found(client: TestClient):
    response = client.get("/api/experiments/does-not-exist/status")
    assert response.status_code == 404
