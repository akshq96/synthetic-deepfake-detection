from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient


def _write_results_file(artifacts_root: Path, results_dir: str, filename: str, data: dict) -> None:
    path = artifacts_root / results_dir / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def test_get_robustness_results(client: TestClient, monkeypatch):
    from backend.app.core.config import settings

    _write_results_file(
        settings.artifacts_root, "experiments/robustness/run1", "robustness_results.json", {"baseline_metrics": {"accuracy": 0.9}}
    )
    response = client.get("/api/results/robustness", params={"results_dir": "experiments/robustness/run1"})
    assert response.status_code == 200
    assert response.json()["baseline_metrics"]["accuracy"] == 0.9


def test_get_results_missing_file_404(client: TestClient):
    response = client.get("/api/results/robustness", params={"results_dir": "experiments/robustness/nope"})
    assert response.status_code == 404


def test_get_results_rejects_path_traversal(client: TestClient):
    response = client.get("/api/results/robustness", params={"results_dir": "../../etc"})
    assert response.status_code == 400


def test_get_ablation_results(client: TestClient):
    from backend.app.core.config import settings

    _write_results_file(
        settings.artifacts_root, "experiments/ablation/run1", "ablation_results.json", [{"point": {}, "eval_metrics": {}}]
    )
    response = client.get("/api/results/ablation", params={"results_dir": "experiments/ablation/run1"})
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_get_unseen_manipulation_results(client: TestClient):
    from backend.app.core.config import settings

    _write_results_file(
        settings.artifacts_root,
        "experiments/unseen_manipulation/run1",
        "unseen_manipulation_results.json",
        {"typeA": {"skipped": False}},
    )
    response = client.get(
        "/api/results/unseen-manipulation", params={"results_dir": "experiments/unseen_manipulation/run1"}
    )
    assert response.status_code == 200
    assert response.json()["typeA"]["skipped"] is False


def test_get_cross_dataset_results(client: TestClient):
    from backend.app.core.config import settings

    _write_results_file(
        settings.artifacts_root,
        "experiments/cross_dataset/run1",
        "cross_dataset_results.json",
        {"train_dataset": "a", "eval_dataset": "b"},
    )
    response = client.get("/api/results/cross-dataset", params={"results_dir": "experiments/cross_dataset/run1"})
    assert response.status_code == 200
    assert response.json()["train_dataset"] == "a"
