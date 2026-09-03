from __future__ import annotations

import numpy as np
import pytest

from ml.evaluation.metrics import compute_metrics, confusion_counts


def test_compute_metrics_perfect_predictions():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.01, 0.02, 0.98, 0.99])
    m = compute_metrics(y_true, y_prob)
    assert m.accuracy == 1.0
    assert m.precision == 1.0
    assert m.recall == 1.0
    assert m.f1 == 1.0
    assert m.roc_auc == 1.0
    assert m.pr_auc == 1.0
    assert m.n_samples == 4


def test_compute_metrics_hand_computed():
    # 2 real (0) misclassified as fake, 2 fake (1) correctly classified.
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.9, 0.9, 0.9, 0.9])
    m = compute_metrics(y_true, y_prob)
    # all predicted fake(1): TP=2, FP=2, FN=0, TN=0
    assert m.accuracy == 0.5
    assert m.precision == 0.5
    assert m.recall == 1.0
    assert m.f1 == pytest.approx(2 / 3)


def test_compute_metrics_mismatched_shapes_raise():
    with pytest.raises(ValueError, match="shape"):
        compute_metrics(np.array([0, 1]), np.array([0.1, 0.2, 0.3]))


def test_compute_metrics_single_class_returns_nan_auc():
    y_true = np.array([0, 0, 0])
    y_prob = np.array([0.1, 0.2, 0.3])
    m = compute_metrics(y_true, y_prob)
    assert np.isnan(m.roc_auc)
    assert np.isnan(m.pr_auc)
    assert m.accuracy == 1.0  # all correctly predicted "real"


def test_confusion_counts_hand_computed():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.9, 0.9, 0.9, 0.9])
    counts = confusion_counts(y_true, y_prob)
    assert counts == {"tn": 0, "fp": 2, "fn": 0, "tp": 2}
