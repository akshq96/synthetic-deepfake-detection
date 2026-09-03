"""Checkpoint save/load/resume.

Layout under `artifacts/checkpoints/<run_id>/`:
  - `latest.pt`  — overwritten every `save_every_steps`, used for `--resume auto`
  - `best.pt`    — overwritten whenever val metric improves
  - `config.yaml` — resolved config snapshot (so a Colab CLI override is still reproducible)

On Colab, `artifacts/` should be symlinked into Google Drive (see
ml/scripts/colab_setup.py, added when Colab docs are written) so a session
disconnect doesn't lose progress — `latest.pt` is what `--resume auto` picks up.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn


@dataclass
class CheckpointState:
    model_state: dict
    optimizer_state: dict
    scheduler_state: dict | None
    epoch: int
    global_step: int
    best_metric: float


def checkpoint_dir(artifacts_root: str | Path, run_id: str) -> Path:
    return Path(artifacts_root) / "checkpoints" / run_id


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler=None,
    epoch: int,
    global_step: int,
    best_metric: float,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "epoch": epoch,
            "global_step": global_step,
            "best_metric": best_metric,
        },
        path,
    )


def load_checkpoint(path: str | Path, *, map_location: str = "cpu") -> CheckpointState:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found at {path}")
    data = torch.load(path, map_location=map_location, weights_only=False)
    return CheckpointState(
        model_state=data["model_state"],
        optimizer_state=data["optimizer_state"],
        scheduler_state=data.get("scheduler_state"),
        epoch=data["epoch"],
        global_step=data["global_step"],
        best_metric=data["best_metric"],
    )


def find_latest_checkpoint(artifacts_root: str | Path, run_id: str) -> Path | None:
    path = checkpoint_dir(artifacts_root, run_id) / "latest.pt"
    return path if path.exists() else None
