"""Gaussian blur at a configurable severity (sigma). See gaussian_noise.py
for why this reuses the SyntheticTechnique interface without being part of
the training-time synthetic-manipulation technique set.
"""

from __future__ import annotations

import numpy as np

from ml.synthetic.base import SyntheticTechnique


class GaussianBlurTechnique(SyntheticTechnique):
    name = "gaussian_blur"

    def __init__(self, sigma: float = 2.0):
        self._sigma = sigma

    def apply(
        self, face_img: np.ndarray, *, donor_img: np.ndarray | None = None, seed: int = 0
    ) -> np.ndarray:
        import cv2

        if self._sigma <= 0:
            return face_img.copy()
        # kernel size must be odd and large enough for the given sigma.
        k = max(3, int(2 * round(3 * self._sigma) + 1))
        return cv2.GaussianBlur(face_img, (k, k), sigmaX=self._sigma).astype(np.uint8)
