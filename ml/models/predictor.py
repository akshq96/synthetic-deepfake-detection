"""Inference wrapper: raw logits -> calibrated probability -> label + abstain.

`abstain_margin` defines a symmetric band around 0.5 in calibrated-probability
space; predictions landing inside it are flagged `abstained=True` rather than
forced to a label. The margin itself should be chosen via a val-set sweep
(accuracy-on-non-abstained vs abstain-rate trade-off) — see
`sweep_abstain_margin` — rather than picked arbitrarily.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from ml.data_pipeline.schema import LABEL_FAKE, LABEL_REAL
from ml.training.calibrate import Calibration


@dataclass(frozen=True)
class Prediction:
    label: str  # LABEL_REAL, LABEL_FAKE, or "abstain"
    confidence: float  # calibrated probability of the predicted class (or of "fake" if abstained)
    fake_probability: float  # calibrated P(fake), always populated
    abstained: bool


class Predictor:
    def __init__(
        self,
        model: nn.Module,
        *,
        calibration: Calibration | None = None,
        abstain_margin: float = 0.1,
        device: str | torch.device = "cpu",
    ):
        self._model = model.to(device).eval()
        self._calibration = calibration or Calibration.identity()
        self._abstain_margin = abstain_margin
        self._device = device

    @torch.no_grad()
    def predict_batch(self, images: torch.Tensor) -> list[Prediction]:
        images = images.to(self._device)
        logits = self._model(images).squeeze(-1).cpu().numpy()
        probs = self._calibration.apply(logits)
        return [self._to_prediction(p) for p in probs]

    def _to_prediction(self, fake_prob: float) -> Prediction:
        low = 0.5 - self._abstain_margin
        high = 0.5 + self._abstain_margin
        if low < fake_prob < high:
            return Prediction(
                label="abstain", confidence=float(fake_prob), fake_probability=float(fake_prob), abstained=True
            )
        label = LABEL_FAKE if fake_prob >= 0.5 else LABEL_REAL
        confidence = fake_prob if label == LABEL_FAKE else 1.0 - fake_prob
        return Prediction(
            label=label, confidence=float(confidence), fake_probability=float(fake_prob), abstained=False
        )


def sweep_abstain_margin(
    val_fake_probs: np.ndarray, val_labels: np.ndarray, *, margins: list[float] | None = None
) -> list[dict]:
    """For each candidate margin, compute abstain-rate and accuracy restricted
    to non-abstained predictions. Returns a list of dicts (one per margin),
    ready to log to MLflow as a table/artifact — this is the val-set sweep
    that should drive the `abstain_margin` config value, not a hardcoded guess.
    """
    margins = margins if margins is not None else [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]
    val_fake_probs = np.asarray(val_fake_probs, dtype=float)
    val_labels = np.asarray(val_labels, dtype=int)

    results = []
    for margin in margins:
        low, high = 0.5 - margin, 0.5 + margin
        abstained_mask = (val_fake_probs > low) & (val_fake_probs < high)
        kept_mask = ~abstained_mask
        n_kept = int(kept_mask.sum())
        if n_kept == 0:
            accuracy = float("nan")
        else:
            preds = (val_fake_probs[kept_mask] >= 0.5).astype(int)
            accuracy = float((preds == val_labels[kept_mask]).mean())
        results.append(
            {
                "margin": margin,
                "abstain_rate": float(abstained_mask.mean()),
                "accuracy_on_non_abstained": accuracy,
                "n_kept": n_kept,
            }
        )
    return results
