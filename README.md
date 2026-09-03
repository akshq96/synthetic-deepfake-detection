# Synthetic Data-Augmented Deepfake Detection

Research system for detecting deepfake images/videos, built to answer a specific
research question empirically:

> **Does synthetic data augmentation improve the generalization of deepfake detectors
> to unseen manipulation techniques?**

The system trains and compares CNN (EfficientNetV2 / ConvNeXt) and Vision Transformer
detectors, with a configurable synthetic-manipulation data pipeline, leakage-safe
train/val/test splits, calibrated confidence + abstain predictions, Grad-CAM/attention
explainability, video support, and a FastAPI + Next.js application for running and
inspecting experiments. Results are never assumed — every claim must trace back to a
logged MLflow run; see `docs/methodology.md` for how that's enforced end to end.

## Status

Built incrementally, phase by phase. Phases 1–11 (dataset pipeline through final
testing/documentation) are implemented and tested end to end on synthetic fixture
data. **No real dataset has been trained on yet** — that (and the genuine Colab/Kaggle
GPU run + real experiment results it enables) is the next step, and is explicitly out
of scope for this repository to fabricate; see `docs/reproducibility.md`.

## Repository layout

- `ml/` — research core: dataset prep, synthetic generation, training, evaluation, XAI,
  video processing. Framework-agnostic, importable by the backend, never the reverse.
- `backend/` — FastAPI app (SQLAlchemy, Alembic, MLflow integration).
- `frontend/` — Next.js + TypeScript + Tailwind dashboard.
- `data/` — datasets (gitignored; see `docs/dataset_setup.md`).
- `artifacts/` — checkpoints, MLflow store, generated reports/figures (gitignored).
- `docs/` — methodology, reproducibility, and setup guides (see below).

## Quickstart

```bash
make install   # venv + pip install -e ".[full]" + frontend npm install
make test-ml   # ml/ pytest suite (dataset pipeline -> XAI -> video), fixture-based
make test-backend
```

Or by hand:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[full]"
pytest ml/tests backend/tests
```

This runs the full pipeline — dataset manifest building, leakage-safe splitting,
CNN/ViT training, synthetic augmentation, generalization/robustness experiments,
XAI, video processing, and the backend API — against a tiny synthetic fixture
dataset generated on the fly. It confirms the pipeline is intact; it does not
produce a research result (there's no real data to train on yet).

Training a real model (once you have a dataset — see `docs/dataset_setup.md`) runs
identically on a CPU debug subset locally or a free Colab/Kaggle GPU via the same
script + config:

```bash
python -m ml.training.train --config ml/configs/cnn_baseline_debug.yaml
```

## Running the application

```bash
make migrate                              # alembic upgrade head
cp .env.example .env                      # set DEFAULT_CHECKPOINT_PATH once you have one
make backend                              # uvicorn, with --reload

# second terminal
cd frontend && npm install && cp .env.local.example .env.local
make frontend                             # next dev
```

The frontend expects the backend at `http://localhost:8000` by default (see
`frontend/.env.local.example`). The `/api/detect/*` endpoints return `503` until
`DEFAULT_CHECKPOINT_PATH` points at a real trained checkpoint — by design, the app
never serves predictions from an untrained/random model without saying so.
`docs/api_reference.md` maps every endpoint; the interactive OpenAPI docs are at
`/docs` once the backend is running.

## Documentation

- [`docs/methodology.md`](docs/methodology.md) — research design, leakage
  prevention, what each synthetic technique approximates (and the compression-
  robustness caveat), what "unseen manipulation" means, ablation design,
  calibration/abstention, why MediaPipe and why the backend's job-execution model
  is what it is.
- [`docs/reproducibility.md`](docs/reproducibility.md) — seed policy, config
  versioning, and the exact sequence to reproduce a real experiment once you have
  data.
- [`docs/dataset_setup.md`](docs/dataset_setup.md) — how to obtain FaceForensics++,
  Celeb-DF v2, and DFDC.
- [`docs/colab_setup.md`](docs/colab_setup.md) / [`docs/kaggle_setup.md`](docs/kaggle_setup.md)
  — free-GPU training setup, session-limit/resume instructions.
- [`docs/api_reference.md`](docs/api_reference.md) — backend endpoint map.

## Testing

```bash
make test-ml        # ml/ pytest suite
make test-backend    # backend/ pytest suite (FastAPI TestClient + throwaway SQLite)
make test-frontend    # vitest component tests
make test-e2e          # Playwright — navigation always runs; the live detect
                        # happy-path test auto-skips if no backend is reachable
```

## License / data notice

This repository does not ship any deepfake dataset. `ml/scripts/download_datasets.py`
validates local dataset presence and prints setup/registration instructions for
FaceForensics++, Celeb-DF v2, and DFDC rather than downloading them automatically
(these require accepting dataset EULAs).
