"""Shared inference/evaluation helpers, used by training (val-loop logging),
and by every experiment script (unseen-manipulation, cross-dataset,
robustness) so "run a checkpoint over a manifest slice and get metrics" is
implemented exactly once.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from ml.datasets.image_dataset import ManifestImageDataset
from ml.evaluation.metrics import Metrics, compute_metrics
from ml.models.factory import build_model
from ml.training.calibrate import Calibration
from ml.training.checkpoint import load_checkpoint


@torch.no_grad()
def collect_logits(
    model: nn.Module, loader: DataLoader, *, device: str | torch.device = "cpu"
) -> tuple[np.ndarray, np.ndarray]:
    """Run `model` over every batch in `loader` in eval mode. Returns
    (logits, labels) as 1-D numpy arrays, in loader iteration order.
    """
    model.eval()
    all_logits, all_labels = [], []
    for batch in loader:
        images = batch["image"].to(device)
        logits = model(images).squeeze(-1).cpu().numpy()
        all_logits.append(logits)
        all_labels.append(batch["label"].numpy())
    return np.concatenate(all_logits), np.concatenate(all_labels)


def load_trained_model(
    checkpoint_path: str | Path, model_cfg, *, device: str | torch.device = "cpu"
) -> nn.Module:
    model = build_model(model_cfg)
    state = load_checkpoint(checkpoint_path, map_location=str(device))
    model.load_state_dict(state.model_state)
    model.to(device).eval()
    return model


def evaluate_on_manifest(
    model: nn.Module,
    manifest: pd.DataFrame,
    *,
    image_size: int,
    calibration: Calibration | None = None,
    batch_size: int = 16,
    device: str | torch.device = "cpu",
) -> tuple[Metrics, dict]:
    """Evaluate `model` on every row of `manifest` (caller is responsible for
    having already filtered it to the desired split/subset — this function
    does not know about splits). Returns (Metrics, raw_predictions_dict)
    where raw_predictions_dict has sample_id/label/logit/prob arrays, useful
    for plotting (confusion matrix, ROC/PR curves) without rerunning
    inference.
    """
    if len(manifest) == 0:
        raise ValueError("evaluate_on_manifest received an empty manifest slice")

    dataset = ManifestImageDataset(manifest, image_size=image_size, train=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    logits, labels = collect_logits(model, loader, device=device)

    calibration = calibration or Calibration.identity()
    probs = calibration.apply(logits)
    metrics = compute_metrics(labels, probs)

    raw = {
        "sample_id": dataset.manifest["sample_id"].tolist(),
        "label": labels.tolist(),
        "logit": logits.tolist(),
        "prob_fake": probs.tolist(),
    }
    return metrics, raw


def load_calibration_if_exists(path: str | Path) -> Calibration:
    path = Path(path)
    return Calibration.load(path) if path.exists() else Calibration.identity()
