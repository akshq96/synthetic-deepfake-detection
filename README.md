# Synthetic Data-Augmented Deepfake Detection

Research system for detecting deepfake images/videos, built to answer a specific
research question empirically:

> **Does synthetic data augmentation improve the generalization of deepfake detectors
> to unseen manipulation techniques?**

The system trains and compares CNN (EfficientNetV2 / ConvNeXt) and Vision Transformer
detectors, with a configurable synthetic-manipulation data pipeline, leakage-safe
train/val/test splits, calibrated confidence + abstain predictions, Grad-CAM/attention
explainability, video support, and a FastAPI + Next.js application for running and
inspecting experiments. Results are never assumed — every claim in the eventual report
must trace back to a logged MLflow run.

## Status

Built incrementally, phase by phase (see `docs/methodology.md` once written). Current
phase: **1 — dataset pipeline**.

## Repository layout

- `ml/` — research core: dataset prep, synthetic generation, training, evaluation, XAI,
  video processing. Framework-agnostic, importable by the backend, never the reverse.
- `backend/` — FastAPI app (SQLAlchemy + PostgreSQL, MLflow integration).
- `frontend/` — Next.js + TypeScript + Tailwind dashboard.
- `data/` — datasets (gitignored; see `docs/dataset_setup.md` once written).
- `artifacts/` — checkpoints, MLflow store, generated reports/figures (gitignored).
- `docs/` — methodology, reproducibility, and setup guides.

## Quickstart (dataset pipeline, Phase 1)

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .
pytest ml/tests
```

Training (later phases) is designed to run identically on a CPU debug subset locally
or on a free Colab/Kaggle GPU via the same script + config, e.g.:

```bash
python -m ml.training.train --config ml/configs/cnn_baseline_debug.yaml
```

## License / data notice

This repository does not ship any deepfake dataset. `ml/scripts/download_datasets.py`
validates local dataset presence and prints setup/registration instructions for
FaceForensics++, Celeb-DF v2, and DFDC rather than downloading them automatically
(these require accepting dataset EULAs).
