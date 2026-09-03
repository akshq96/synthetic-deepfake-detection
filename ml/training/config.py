"""Config loading (OmegaConf/YAML) with CLI dotlist overrides, e.g.:

    python -m ml.training.train --config ml/configs/cnn_baseline_debug.yaml \
        training.lr=1e-4 data.synthetic_ratio=0.3

Two config libraries are used deliberately in this repo for different layers:
OmegaConf here (supports CLI overrides, useful for ablation sweeps), Pydantic
in the backend (request/response validation) — see docs/methodology.md once
written.
"""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_config(config_path: str | Path, cli_overrides: list[str] | None = None) -> DictConfig:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"config not found: {config_path}")
    cfg = OmegaConf.load(config_path)
    if cli_overrides:
        override_cfg = OmegaConf.from_dotlist(cli_overrides)
        cfg = OmegaConf.merge(cfg, override_cfg)
    return cfg  # type: ignore[return-value]


def flatten_for_logging(cfg: DictConfig, *, sep: str = ".") -> dict:
    """Flatten a nested config into dotted-key -> scalar pairs, for
    mlflow.log_params (which requires a flat dict of primitives).
    """
    container = OmegaConf.to_container(cfg, resolve=True)
    flat: dict = {}

    def _walk(prefix: str, value) -> None:
        if isinstance(value, dict):
            for k, v in value.items():
                _walk(f"{prefix}{sep}{k}" if prefix else str(k), v)
        elif isinstance(value, list):
            flat[prefix] = ",".join(str(v) for v in value)
        else:
            flat[prefix] = value

    _walk("", container)
    return flat
