from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class RunOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    experiment_id: str
    run_name: str
    mlflow_run_id: str | None
    status: str
    pid: int | None
    error_message: str | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None


class ExperimentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    description: str | None
    created_at: datetime
    runs: list[RunOut] = []


class SyntheticLabRunRequest(BaseModel):
    """Configures and launches one Synthetic Data Lab training run — the
    "baseline vs synthetic-augmented" comparison the frontend's lab view is
    built around.
    """

    experiment_name: str = Field(..., min_length=1, max_length=255)
    run_name: str = Field(..., min_length=1, max_length=255)
    base_config_path: str = Field(
        default="ml/configs/cnn_baseline_debug.yaml",
        description="Path (relative to repo root) to the base training config to override.",
    )
    model_name: str | None = Field(default=None, description="Overrides model.name, e.g. 'vit_base_patch16_224'")
    synthetic_ratio: float = Field(default=0.0, ge=0.0, lt=1.0)
    synthetic_techniques: list[str] = Field(default_factory=list)
    manifest_path: str | None = Field(default=None, description="Overrides data.manifest_path")
    epochs: int | None = Field(default=None, ge=1)


class RunStatusOut(BaseModel):
    id: str
    status: str
    pid: int | None
    error_message: str | None
    mlflow_run_id: str | None
