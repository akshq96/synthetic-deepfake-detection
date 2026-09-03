"""Additive Gaussian pixel noise at a configurable severity (std-dev in
0-255 pixel units). Used by the robustness experiment (Phase 6) as a
perturbation applied at evaluation time, independent of the 5 synthetic-
manipulation *training* techniques (base.py's SyntheticTechnique interface
is reused here purely because the seeded-determinism contract is identical,
not because this is meant as a training-augmentation technique).
"""

from __future__ import annotations

import numpy as np

from ml.synthetic.base import SyntheticTechnique


class GaussianNoiseTechnique(SyntheticTechnique):
    name = "gaussian_noise"

    def __init__(self, sigma: float = 15.0):
        self._sigma = sigma

    def apply(
        self, face_img: np.ndarray, *, donor_img: np.ndarray | None = None, seed: int = 0
    ) -> np.ndarray:
        rng = np.random.default_rng(seed)
        noise = rng.normal(loc=0.0, scale=self._sigma, size=face_img.shape)
        return np.clip(face_img.astype(np.float64) + noise, 0, 255).astype(np.uint8)
