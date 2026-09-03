"""Temperature scaling (Guo et al. 2017): a single learnable scalar T that
rescales logits post-hoc so predicted probabilities better match empirical
accuracy, fit on the validation set only (never train or test).

calibrated_prob = sigmoid(logit / T)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class Calibration:
    temperature: float

    def apply(self, logits: np.ndarray) -> np.ndarray:
        from scipy.special import expit

        logits = np.asarray(logits, dtype=np.float64)
        return expit(logits / self.temperature)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"temperature": self.temperature}))

    @classmethod
    def load(cls, path: str | Path) -> "Calibration":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"calibration file not found at {path}; run ml.training.calibrate "
                f"on a trained checkpoint first."
            )
        data = json.loads(path.read_text())
        return cls(temperature=float(data["temperature"]))

    @classmethod
    def identity(cls) -> "Calibration":
        """T=1.0 — no rescaling. Used as a fallback when no calibration file
        exists yet (e.g. mid-training), never silently substituted once a
        real calibration has been fit.
        """
        return cls(temperature=1.0)


def fit_temperature(
    val_logits: np.ndarray, val_labels: np.ndarray, *, max_iter: int = 50, lr: float = 0.01
) -> Calibration:
    """Fit T on validation logits via LBFGS minimizing BCE-with-logits, as in
    Guo et al. 2017. `val_logits`/`val_labels` are 1-D arrays (pre-sigmoid
    logits and 0/1 labels respectively).
    """
    logits_t = torch.tensor(np.asarray(val_logits, dtype=np.float32))
    labels_t = torch.tensor(np.asarray(val_labels, dtype=np.float32))

    log_temperature = nn.Parameter(torch.zeros(1))  # T = exp(0) = 1.0 initially
    optimizer = torch.optim.LBFGS([log_temperature], lr=lr, max_iter=max_iter)
    criterion = nn.BCEWithLogitsLoss()

    def closure():
        optimizer.zero_grad()
        temperature = torch.exp(log_temperature)
        loss = criterion(logits_t / temperature, labels_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = float(torch.exp(log_temperature).item())
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError(f"temperature fit diverged to a non-positive/non-finite value: {temperature}")
    return Calibration(temperature=temperature)
