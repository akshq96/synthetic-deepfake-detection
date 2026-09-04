from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.app.api import detect, experiments, mlflow_proxy, models, models_compare, reports, results
from backend.app.core.config import settings
from backend.app.db.base import create_all_tables


@asynccontextmanager
async def lifespan(app: FastAPI):
    for path in (settings.uploads_dir, settings.heatmaps_dir, settings.reports_dir, settings.experiments_dir):
        path.mkdir(parents=True, exist_ok=True)
    create_all_tables()  # dev convenience; real deployments run `alembic upgrade head` instead
    yield


app = FastAPI(
    title="Synthetic Data-Augmented Deepfake Detection API",
    description=(
        "Backend for the deepfake detection research system: image/video detection, "
        "synthetic-augmentation experiments, and generalization/robustness results."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(detect.router)
app.include_router(experiments.router)
app.include_router(results.router)
app.include_router(models_compare.router)
app.include_router(models.router)
app.include_router(reports.router)
app.include_router(mlflow_proxy.router)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


# Serves uploaded images/videos, heatmap overlays, and generated reports —
# DB rows store paths relative to artifacts_root, so `/static/<that path>`
# always resolves correctly regardless of where artifacts_root is mounted.
settings.artifacts_root.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(settings.artifacts_root)), name="static")
