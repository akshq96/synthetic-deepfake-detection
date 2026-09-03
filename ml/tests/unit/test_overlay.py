from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ml.xai.overlay import render_overlay, save_overlay


def _sample_image() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)


def test_render_overlay_shape_matches_original():
    img = _sample_image()
    heatmap = np.random.default_rng(1).random((64, 64)).astype(np.float32)
    overlay = render_overlay(img, heatmap)
    assert overlay.shape == img.shape
    assert overlay.dtype == np.uint8


def test_render_overlay_resizes_mismatched_heatmap():
    img = _sample_image()
    heatmap = np.random.default_rng(1).random((16, 16)).astype(np.float32)
    overlay = render_overlay(img, heatmap)
    assert overlay.shape == img.shape


def test_render_overlay_handles_uniform_heatmap():
    img = _sample_image()
    heatmap = np.full((64, 64), 0.5, dtype=np.float32)
    overlay = render_overlay(img, heatmap)  # must not raise (div-by-zero guard)
    assert overlay.shape == img.shape


def test_render_overlay_rejects_non_2d_heatmap():
    img = _sample_image()
    heatmap = np.zeros((64, 64, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="must be 2-D"):
        render_overlay(img, heatmap)


def test_save_overlay_writes_file(tmp_path: Path):
    img = _sample_image()
    heatmap = np.random.default_rng(1).random((64, 64)).astype(np.float32)
    out_path = save_overlay(img, heatmap, tmp_path / "heatmaps" / "out.png")
    assert out_path.exists()
    assert out_path.stat().st_size > 0
