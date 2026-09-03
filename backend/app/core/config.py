"""Application settings, loaded from environment variables / .env.

DATABASE_URL defaults to a local SQLite file so the backend runs with zero
setup for local/solo-dev use (matches this project's local-first scope
decision — see docs/methodology.md once written); set it to a
postgresql+psycopg://... URL for a real Postgres deployment, which the
SQLAlchemy models are written to be compatible with (no SQLite-only types).
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str = f"sqlite:///{REPO_ROOT / 'artifacts' / 'app.db'}"
    artifacts_root: Path = REPO_ROOT / "artifacts"
    # Deliberately NOT named/aliased to the plain "MLFLOW_TRACKING_URI" env
    # var: mlflow.set_tracking_uri() (called by ml.training.train for every
    # run) sets that exact env var as a side effect. If this field read it
    # too, any process that ever trains a model would permanently poison
    # this setting for the rest of its lifetime with wherever training last
    # pointed MLflow — a real bug caught by cross-test pollution in the test
    # suite (an ml/ smoke test running before a backend test broke the
    # backend's own MLflow queries). "APP_MLFLOW_TRACKING_URI" is a distinct,
    # explicit override for the *backend's* default, independent of whatever
    # ml.training.train's own process-local mlflow client is pointed at.
    mlflow_tracking_uri: str | None = Field(default=None, validation_alias="APP_MLFLOW_TRACKING_URI")
    cors_allow_origins: list[str] = ["http://localhost:3000"]
    max_upload_size_bytes: int = 200 * 1024 * 1024  # 200 MB

    # No checkpoint is bundled with this repo (no real dataset has been
    # trained on yet — see README/docs/dataset_setup.md). Detect endpoints
    # refuse to run rather than silently serving predictions from an
    # untrained/random model, per the project's "never fabricate results"
    # principle extended to the application layer. Set these once you have
    # a real trained checkpoint (e.g. from ml.training.train).
    default_checkpoint_path: str | None = None
    default_calibration_path: str | None = None
    default_model_name: str = "efficientnetv2_s"
    default_image_size: int = 224

    @property
    def uploads_dir(self) -> Path:
        return self.artifacts_root / "uploads"

    @property
    def heatmaps_dir(self) -> Path:
        return self.artifacts_root / "heatmaps"

    @property
    def reports_dir(self) -> Path:
        return self.artifacts_root / "reports"

    @property
    def experiments_dir(self) -> Path:
        return self.artifacts_root / "experiments"

    @property
    def resolved_mlflow_tracking_uri(self) -> str:
        return self.mlflow_tracking_uri or f"sqlite:///{self.artifacts_root / 'mlflow.db'}"


settings = Settings()
