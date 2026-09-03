"""Phase 7 smoke test: get_explainer(model_type).explain(...) returns a valid
heatmap for both a CNN and a ViT fixture checkpoint, saved as an overlay PNG
— the definition of done for Phase 7 per the project plan.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from ml.data_pipeline.manifest import save_manifest
from ml.data_pipeline.split import assign_splits
from ml.evaluation.evaluate import load_trained_model
from ml.tests.fixtures.synthetic_fixture import generate_fixture_dataset
from ml.training.config import load_config
from ml.training.train import run_training
from ml.xai.factory import get_explainer
from ml.xai.overlay import save_overlay


def _train_debug_checkpoint(tmp_path: Path, config_path: str, run_name: str):
    raw = generate_fixture_dataset(tmp_path / "raw")
    split = assign_splits(raw, seed=42)
    manifest_path = tmp_path / "manifests" / "fixture_split.parquet"
    save_manifest(split, manifest_path)

    artifacts_root = tmp_path / "artifacts"
    cfg = load_config(
        Path(config_path),
        cli_overrides=[
            f"data.manifest_path={manifest_path}",
            f"artifacts_root={artifacts_root}",
            f"run_name={run_name}",
        ],
    )
    summary = run_training(cfg)
    return cfg, summary


def test_gradcam_explain_and_overlay_on_trained_cnn(tmp_path: Path):
    cfg, summary = _train_debug_checkpoint(tmp_path, "ml/configs/cnn_baseline_debug.yaml", "xai_cnn_smoke")
    model = load_trained_model(summary["checkpoint_path"], cfg.model)

    explainer = get_explainer(cfg.model.name)
    image_size = int(cfg.data.image_size)
    x = torch.randn(1, 3, image_size, image_size)
    heatmap = explainer.explain(model, x, target_class=1)
    assert heatmap.shape == (1, image_size, image_size)

    original = np.random.default_rng(0).integers(0, 256, size=(image_size, image_size, 3), dtype=np.uint8)
    out_path = save_overlay(original, heatmap[0], tmp_path / "heatmaps" / "cnn_overlay.png")
    assert out_path.exists()
    assert cv2.imread(str(out_path)) is not None


def test_attention_rollout_explain_and_overlay_on_trained_vit(tmp_path: Path):
    cfg, summary = _train_debug_checkpoint(tmp_path, "ml/configs/vit_baseline_debug.yaml", "xai_vit_smoke")
    model = load_trained_model(summary["checkpoint_path"], cfg.model)

    explainer = get_explainer(cfg.model.name)
    image_size = int(cfg.data.image_size)
    x = torch.randn(1, 3, image_size, image_size)
    heatmap = explainer.explain(model, x)
    assert heatmap.shape == (1, image_size, image_size)

    original = np.random.default_rng(0).integers(0, 256, size=(image_size, image_size, 3), dtype=np.uint8)
    out_path = save_overlay(original, heatmap[0], tmp_path / "heatmaps" / "vit_overlay.png")
    assert out_path.exists()
    assert cv2.imread(str(out_path)) is not None
