"""Read-only endpoints serving already-computed experiment results (JSON
files written by ml/evaluation/experiments/*.py and ml/evaluation/ablation.py)
— no recomputation happens server-side, per the project plan's evaluation
design; these scripts are run via the CLI (or, for training-based ones,
launched like any other job — see job_launcher.py) and their output
directory is passed to these endpoints to read back.

`results_dir` is relative to `artifacts_root` (e.g.
"experiments/unseen_manipulation/2026-01-01T00-00-00"), never an absolute
or `..`-escaping path — enforced below.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

from backend.app.core.config import settings

router = APIRouter(prefix="/api/results", tags=["results"])

RESULT_FILENAMES = {
    "baseline-vs-synthetic": "ablation_results.json",
    "unseen-manipulation": "unseen_manipulation_results.json",
    "cross-dataset": "cross_dataset_results.json",
    "robustness": "robustness_results.json",
    "ablation": "ablation_results.json",
}


def _resolve_results_dir(results_dir: str) -> Path:
    resolved = (settings.artifacts_root / results_dir).resolve()
    artifacts_root_resolved = settings.artifacts_root.resolve()
    if artifacts_root_resolved not in resolved.parents and resolved != artifacts_root_resolved:
        raise HTTPException(status_code=400, detail="results_dir must stay within the artifacts directory")
    return resolved


def _read_results(results_dir: str, filename: str) -> dict | list:
    path = _resolve_results_dir(results_dir) / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"no results file found at {results_dir}/{filename}")
    return json.loads(path.read_text())


# baseline-vs-synthetic and ablation both read ablation_results.json, which
# ml.evaluation.ablation.run_ablation_sweep produces as a JSON *list* (one
# entry per grid point) — unlike the other three experiment scripts, which
# each write a single JSON object.
@router.get("/baseline-vs-synthetic")
def get_baseline_vs_synthetic(results_dir: str = Query(...)) -> list:
    return _read_results(results_dir, RESULT_FILENAMES["baseline-vs-synthetic"])


@router.get("/unseen-manipulation")
def get_unseen_manipulation(results_dir: str = Query(...)) -> dict:
    return _read_results(results_dir, RESULT_FILENAMES["unseen-manipulation"])


@router.get("/cross-dataset")
def get_cross_dataset(results_dir: str = Query(...)) -> dict:
    return _read_results(results_dir, RESULT_FILENAMES["cross-dataset"])


@router.get("/robustness")
def get_robustness(results_dir: str = Query(...)) -> dict:
    return _read_results(results_dir, RESULT_FILENAMES["robustness"])


@router.get("/ablation")
def get_ablation(results_dir: str = Query(...)) -> list:
    return _read_results(results_dir, RESULT_FILENAMES["ablation"])
