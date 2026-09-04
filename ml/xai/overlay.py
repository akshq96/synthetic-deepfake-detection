"""Render a heatmap as an alpha-blended overlay on the original (unnormalized,
uint8) face crop, and save it as a PNG — the output convention every caller
(training-time inspection, the API, the frontend) uses, so heatmap image
processing happens exactly once.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def render_heatmap_colormap(heatmap: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """`heatmap`: HxW float array (any range; rescaled to [0,1] here).
    `target_shape`: (h, w) to resize to if the heatmap's spatial size differs.
    Returns an HxWx3 uint8 BGR colormap image — the pure heatmap, with no
    original-image blend (that's `render_overlay`, which calls this).
    """
    import cv2

    if heatmap.ndim != 2:
        raise ValueError(f"heatmap must be 2-D (H, W), got shape {heatmap.shape}")

    h, w = target_shape
    if heatmap.shape != (h, w):
        heatmap = cv2.resize(heatmap.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)

    heatmap = heatmap.astype(np.float64)
    heatmap_range = heatmap.max() - heatmap.min()
    if heatmap_range > 1e-8:
        heatmap = (heatmap - heatmap.min()) / heatmap_range
    else:
        heatmap = np.zeros_like(heatmap)

    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    return cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)


def render_overlay(
    original_image_bgr: np.ndarray,
    heatmap: np.ndarray,
    *,
    alpha: float = 0.5,
) -> np.ndarray:
    """`original_image_bgr`: HxWx3 uint8. `heatmap`: HxW float array (any
    range; rescaled to [0,1] here) — resized to match `original_image_bgr` if
    its spatial size differs. Returns an HxWx3 uint8 BGR overlay image.
    """
    import cv2

    h, w = original_image_bgr.shape[:2]
    heatmap_color = render_heatmap_colormap(heatmap, (h, w))
    overlay = cv2.addWeighted(heatmap_color, alpha, original_image_bgr, 1.0 - alpha, 0.0)
    return overlay


def save_overlay(
    original_image_bgr: np.ndarray, heatmap: np.ndarray, output_path: str | Path, *, alpha: float = 0.5
) -> Path:
    import cv2

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    overlay = render_overlay(original_image_bgr, heatmap, alpha=alpha)
    ok = cv2.imwrite(str(output_path), overlay)
    if not ok:
        raise RuntimeError(f"failed to write overlay image to {output_path}")
    return output_path


def save_heatmap_only(heatmap: np.ndarray, target_shape: tuple[int, int], output_path: str | Path) -> Path:
    """Saves the pure (non-blended) heatmap colormap — the "Heatmap" panel in
    the Detect page's 3-panel Original/Heatmap/Overlay view, alongside the
    existing blended `save_overlay` output."""
    import cv2

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    heatmap_color = render_heatmap_colormap(heatmap, target_shape)
    ok = cv2.imwrite(str(output_path), heatmap_color)
    if not ok:
        raise RuntimeError(f"failed to write heatmap image to {output_path}")
    return output_path


def save_original(image_bgr: np.ndarray, output_path: str | Path) -> Path:
    """Saves the plain original (face-crop) image — the "Original" panel in
    the Detect page's 3-panel view. A thin wrapper over cv2.imwrite so every
    caller goes through this module's error-handling convention."""
    import cv2

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(output_path), image_bgr)
    if not ok:
        raise RuntimeError(f"failed to write original image to {output_path}")
    return output_path
