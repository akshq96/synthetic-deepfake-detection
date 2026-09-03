from __future__ import annotations

import numpy as np
import torch
from torch import nn

from ml.data_pipeline.schema import LABEL_FAKE, LABEL_REAL
from ml.models.predictor import Predictor, sweep_abstain_margin
from ml.training.calibrate import Calibration


class _ConstantLogitModel(nn.Module):
    """Ignores its input entirely and returns a fixed logit per batch
    element (batch size inferred from input) — lets tests drive Predictor's
    abstain/label logic deterministically without a real trained model.
    """

    def __init__(self, logits: list[float]):
        super().__init__()
        self._logits = torch.tensor(logits, dtype=torch.float32)
        self.dummy = nn.Parameter(torch.zeros(1))  # so .to(device) has something to move

    def forward(self, x):
        batch_size = x.shape[0]
        return self._logits[:batch_size].reshape(batch_size, 1)


def test_predictor_labels_confident_predictions():
    # logit=3 -> sigmoid ~0.953 (confidently fake); logit=-3 -> ~0.047 (confidently real)
    model = _ConstantLogitModel([3.0, -3.0])
    predictor = Predictor(model, abstain_margin=0.1)
    preds = predictor.predict_batch(torch.zeros(2, 3, 4, 4))
    assert preds[0].label == LABEL_FAKE
    assert not preds[0].abstained
    assert preds[1].label == LABEL_REAL
    assert not preds[1].abstained


def test_predictor_abstains_near_decision_boundary():
    # logit=0 -> prob=0.5, dead center of the abstain band.
    model = _ConstantLogitModel([0.0])
    predictor = Predictor(model, abstain_margin=0.1)
    preds = predictor.predict_batch(torch.zeros(1, 3, 4, 4))
    assert preds[0].abstained
    assert preds[0].label == "abstain"


def test_predictor_applies_calibration():
    # logit=3 without calibration -> confidently fake; with a large
    # temperature it should be pulled toward 0.5 and abstain instead.
    model = _ConstantLogitModel([3.0])
    predictor = Predictor(model, calibration=Calibration(temperature=20.0), abstain_margin=0.1)
    preds = predictor.predict_batch(torch.zeros(1, 3, 4, 4))
    assert preds[0].abstained


def test_sweep_abstain_margin_monotonic_abstain_rate():
    rng = np.random.default_rng(0)
    probs = rng.random(200)
    labels = (probs >= 0.5).astype(int)
    results = sweep_abstain_margin(probs, labels, margins=[0.0, 0.1, 0.2, 0.3])
    rates = [r["abstain_rate"] for r in results]
    assert rates == sorted(rates)  # wider margin never abstains less


def test_sweep_abstain_margin_perfect_labels_perfect_accuracy():
    probs = np.array([0.9, 0.1, 0.8, 0.2])
    labels = np.array([1, 0, 1, 0])
    results = sweep_abstain_margin(probs, labels, margins=[0.0])
    assert results[0]["accuracy_on_non_abstained"] == 1.0
    assert results[0]["abstain_rate"] == 0.0
