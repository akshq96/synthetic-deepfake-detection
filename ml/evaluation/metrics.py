"""Single source of truth for metric computation, reused by every training
run, experiment script, and the plotting module — so numbers are computed
identically everywhere and are directly comparable across experiments.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class Metrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    n_samples: int

    def to_dict(self) -> dict:
        return asdict(self)


def compute_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, *, threshold: float = 0.5
) -> Metrics:
    """Compute accuracy/precision/recall/F1 (at `threshold`) plus ROC-AUC and
    PR-AUC (threshold-independent, computed from `y_prob` directly).

    y_true: 0/1 array. y_prob: calibrated-or-not probability of the positive
    (fake) class in [0, 1]. Both must be 1-D and the same length.
    """
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    if y_true.shape != y_prob.shape:
        raise ValueError(f"y_true shape {y_true.shape} != y_prob shape {y_prob.shape}")
    if y_true.ndim != 1:
        raise ValueError(f"expected 1-D arrays, got shape {y_true.shape}")

    y_pred = (y_prob >= threshold).astype(int)

    n_classes = len(np.unique(y_true))
    if n_classes < 2:
        # Degenerate case (e.g. a tiny fixture split with only one class
        # present) — AUC is undefined; report NaN rather than raising, so
        # pipeline runs on small fixtures don't crash, but the NaN is visible
        # (never silently substituted with a fake number).
        roc_auc = float("nan")
        pr_auc = float("nan")
    else:
        roc_auc = float(roc_auc_score(y_true, y_prob))
        pr_auc = float(average_precision_score(y_true, y_prob))

    return Metrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        n_samples=int(len(y_true)),
    )


def confusion_counts(y_true: np.ndarray, y_prob: np.ndarray, *, threshold: float = 0.5) -> dict:
    """Return {tn, fp, fn, tp} — the raw confusion-matrix counts, kept
    separate from Metrics since not every caller needs them (plotting does).
    """
    from sklearn.metrics import confusion_matrix

    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_prob).astype(float) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
