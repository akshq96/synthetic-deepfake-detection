from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ml.training.calibrate import Calibration, fit_temperature


def test_calibration_identity_is_noop():
    cal = Calibration.identity()
    logits = np.array([-2.0, 0.0, 2.0])
    probs = cal.apply(logits)
    expected = 1.0 / (1.0 + np.exp(-logits))
    assert np.allclose(probs, expected)


def test_calibration_save_load_roundtrip(tmp_path: Path):
    cal = Calibration(temperature=1.73)
    path = tmp_path / "temperature.json"
    cal.save(path)
    loaded = Calibration.load(path)
    assert loaded.temperature == pytest.approx(1.73)


def test_calibration_load_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="calibration file not found"):
        Calibration.load(tmp_path / "nope.json")


def test_fit_temperature_reduces_overconfidence():
    # Severely overconfident but often-wrong logits: large magnitude, but
    # roughly half the "confident" predictions are actually incorrect.
    rng = np.random.default_rng(0)
    n = 200
    labels = rng.integers(0, 2, size=n).astype(np.float32)
    # Correct-direction large logits for only 60% of samples; wrong-direction
    # large logits for the rest -> overconfident and only 60% accurate.
    correct_mask = rng.random(n) < 0.6
    sign = np.where(labels > 0.5, 1.0, -1.0)
    sign = np.where(correct_mask, sign, -sign)
    logits = sign * 5.0  # large magnitude -> extreme (over)confidence

    cal = fit_temperature(logits, labels)
    # A well-fit temperature for genuinely overconfident logits should be > 1
    # (softening the extreme logits back toward calibrated probabilities).
    assert cal.temperature > 1.0


def test_fit_temperature_rejects_degenerate_result_gracefully():
    # Perfectly separable logits/labels: BCE keeps decreasing as T->0, so
    # LBFGS may drive log_temperature very negative. We only assert the
    # function either returns a valid positive-finite temperature or raises
    # ValueError — it must never return a NaN/negative Calibration silently.
    labels = np.array([0.0, 0.0, 1.0, 1.0] * 20, dtype=np.float32)
    logits = np.array([-10.0, -10.0, 10.0, 10.0] * 20, dtype=np.float32)
    try:
        cal = fit_temperature(logits, labels)
    except ValueError:
        return
    assert cal.temperature > 0 and np.isfinite(cal.temperature)
