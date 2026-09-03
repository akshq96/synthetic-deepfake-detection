# API Reference

The full interactive reference — every endpoint, request/response schema,
and a "try it" console — is generated automatically by FastAPI and served
at **`/docs`** (Swagger UI) or **`/redoc`** whenever the backend is running
(`uvicorn backend.app.main:app`). This page is a short map of what's where
and why, not a substitute for that.

## Detection — `backend/app/api/detect.py`

- `POST /api/detect/image`, `POST /api/detect/video` — upload a file, get a
  `PredictionOut` back (label, calibrated confidence, abstain flag, heatmap
  path; video additionally returns `suspicious_frames`). Runs in-process and
  synchronously (see `docs/methodology.md`'s job-execution-model section).
  Returns `503` if no checkpoint is configured
  (`DEFAULT_CHECKPOINT_PATH` unset) — this app never serves a prediction
  from an untrained/random model silently.
- `GET /api/detect/{prediction_id}` — re-fetch a past prediction.

## Experiments — `backend/app/api/experiments.py`

- `GET /api/experiments`, `GET /api/experiments/{experiment_id}` — list/get
  `Experiment` rows (a named group of training runs) with their nested runs.
- `POST /api/experiments/synthetic-lab/run` — the Synthetic Data Lab's
  launch endpoint: configures a `synthetic_ratio`/`synthetic_techniques` mix
  on top of a base config and launches `ml.training.train` as a subprocess.
  Returns the created `Run` immediately (`status="running"`).
- `GET /api/experiments/{run_id}/status` — poll a `Run`'s status (note:
  `{run_id}` here, not an experiment id — see the module docstring).

## Results — `backend/app/api/results.py`

Read-only passthrough of JSON files already written by
`ml/evaluation/experiments/*.py` / `ml/evaluation/ablation.py` — nothing is
recomputed server-side. Every endpoint takes a `results_dir` query param
(a path relative to `artifacts/`, guarded against escaping it):

- `GET /api/results/baseline-vs-synthetic`, `GET /api/results/ablation` —
  both read an `ablation_results.json` (a JSON *list*, one entry per grid
  point) produced by `ml.evaluation.ablation`.
- `GET /api/results/unseen-manipulation` — reads
  `unseen_manipulation_results.json` from `ml.evaluation.experiments.unseen_manipulation`.
- `GET /api/results/cross-dataset` — reads `cross_dataset_results.json` from
  `ml.evaluation.experiments.cross_dataset`.
- `GET /api/results/robustness` — reads `robustness_results.json` from
  `ml.evaluation.experiments.robustness`.

## Models & MLflow — `models_compare.py`, `mlflow_proxy.py`

- `GET /api/models/compare?experiment_name=...` — the most recent MLflow run
  per architecture (`model.name` param) within an experiment, for the CNN
  vs. ViT view.
- `GET /api/mlflow/runs?experiment_name=...` — thin proxy over
  `MlflowClient.search_runs`, for the experiment-history view.

## Reports — `backend/app/api/reports.py`

- `POST /api/reports/generate` — renders a one-page forensic PDF (label,
  metrics, embedded heatmap) for a given `prediction_id`
  (`backend/app/services/report_service.py`, via `reportlab`).
- `GET /api/reports`, `GET /api/reports/{id}` — list/get report metadata.
- `GET /api/reports/{id}/export` — streams the PDF file
  (`application/pdf`).

## Static files

`/static/<path>` serves everything under `artifacts/` (uploads, heatmap
overlays, generated reports) — every path stored in the database is
relative to `artifacts/`, so `f"{API_BASE_URL}/static/{relative_path}"`
always resolves correctly regardless of where `artifacts_root` is mounted
on disk (see `backend/app/services/storage.py`).
