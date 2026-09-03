from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import torch

from ml.evaluation.evaluate import (
    evaluate_on_manifest,
    load_calibration_if_exists,
    load_trained_model,
)
from ml.models.factory import ModelConfig, build_model
from ml.training.calibrate import Calibration
from ml.training.checkpoint import save_checkpoint


def _save_tiny_checkpoint(tmp_path: Path, model_name: str = "efficientnetv2_s") -> Path:
    model = build_model(ModelConfig(name=model_name, pretrained=False))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    path = tmp_path / "ckpt" / "latest.pt"
    save_checkpoint(path, model=model, optimizer=optimizer, epoch=0, global_step=1, best_metric=0.0)
    return path


def test_load_trained_model_roundtrip(tmp_path: Path):
    ckpt_path = _save_tiny_checkpoint(tmp_path)
    model = load_trained_model(ckpt_path, ModelConfig(name="efficientnetv2_s", pretrained=False))
    assert not model.training  # eval() called
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 1)


def test_evaluate_on_manifest_returns_metrics_and_raw(fixture_manifest_split: pd.DataFrame, tmp_path: Path):
    ckpt_path = _save_tiny_checkpoint(tmp_path)
    model = load_trained_model(ckpt_path, ModelConfig(name="efficientnetv2_s", pretrained=False))

    train_subset = fixture_manifest_split[fixture_manifest_split["split"] == "train"]
    metrics, raw = evaluate_on_manifest(model, train_subset, image_size=32)

    assert metrics.n_samples == len(train_subset)
    assert len(raw["sample_id"]) == len(train_subset)
    assert len(raw["prob_fake"]) == len(train_subset)
    assert all(0.0 <= p <= 1.0 for p in raw["prob_fake"])


def test_evaluate_on_manifest_empty_raises():
    from ml.data_pipeline.schema import empty_manifest

    model = build_model(ModelConfig(name="efficientnetv2_s", pretrained=False))
    with pytest.raises(ValueError, match="empty manifest"):
        evaluate_on_manifest(model, empty_manifest(), image_size=32)


def test_evaluate_on_manifest_applies_calibration(fixture_manifest_split: pd.DataFrame, tmp_path: Path):
    ckpt_path = _save_tiny_checkpoint(tmp_path)
    model = load_trained_model(ckpt_path, ModelConfig(name="efficientnetv2_s", pretrained=False))
    train_subset = fixture_manifest_split[fixture_manifest_split["split"] == "train"].head(2)

    metrics_uncalibrated, raw_uncalibrated = evaluate_on_manifest(model, train_subset, image_size=32)
    metrics_calibrated, raw_calibrated = evaluate_on_manifest(
        model, train_subset, image_size=32, calibration=Calibration(temperature=50.0)
    )
    # A very large temperature must pull every probability toward 0.5.
    for p in raw_calibrated["prob_fake"]:
        assert abs(p - 0.5) < 0.1
    assert raw_uncalibrated["logit"] == raw_calibrated["logit"]  # same model, same logits


def test_load_calibration_if_exists_missing_returns_identity(tmp_path: Path):
    cal = load_calibration_if_exists(tmp_path / "nope.json")
    assert cal.temperature == 1.0


def test_load_calibration_if_exists_loads_real_file(tmp_path: Path):
    path = tmp_path / "temperature.json"
    Calibration(temperature=2.5).save(path)
    cal = load_calibration_if_exists(path)
    assert cal.temperature == 2.5
