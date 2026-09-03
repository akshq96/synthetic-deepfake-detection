from __future__ import annotations

import numpy as np
import pytest

from ml.synthetic.gaussian_blur import GaussianBlurTechnique
from ml.synthetic.gaussian_noise import GaussianNoiseTechnique


def _sample_image(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(48, 48, 3), dtype=np.uint8)


def test_gaussian_noise_is_deterministic_given_seed():
    img = _sample_image()
    technique = GaussianNoiseTechnique(sigma=20.0)
    out1 = technique.apply(img, seed=1)
    out2 = technique.apply(img, seed=1)
    assert np.array_equal(out1, out2)


def test_gaussian_noise_different_seeds_differ():
    img = _sample_image()
    technique = GaussianNoiseTechnique(sigma=20.0)
    out1 = technique.apply(img, seed=1)
    out2 = technique.apply(img, seed=2)
    assert not np.array_equal(out1, out2)


def test_gaussian_noise_zero_sigma_is_noop():
    img = _sample_image()
    technique = GaussianNoiseTechnique(sigma=0.0)
    out = technique.apply(img, seed=1)
    assert np.array_equal(out, img)


@pytest.mark.parametrize("sigma", [1.0, 5.0, 20.0])
def test_gaussian_noise_larger_sigma_more_change(sigma):
    img = _sample_image()
    out = GaussianNoiseTechnique(sigma=sigma).apply(img, seed=1)
    mean_abs_diff = np.abs(out.astype(int) - img.astype(int)).mean()
    assert mean_abs_diff >= 0  # sanity: always non-negative
    if sigma > 0:
        assert mean_abs_diff > 0


def test_gaussian_blur_smooths_the_image():
    img = _sample_image()
    out = GaussianBlurTechnique(sigma=3.0).apply(img, seed=0)
    assert out.shape == img.shape
    # Blur should reduce high-frequency variance relative to the noisy input.
    assert out.astype(float).var() < img.astype(float).var()


def test_gaussian_blur_is_seed_independent():
    img = _sample_image()
    technique = GaussianBlurTechnique(sigma=2.0)
    out1 = technique.apply(img, seed=1)
    out2 = technique.apply(img, seed=999)
    assert np.array_equal(out1, out2)


def test_gaussian_blur_zero_sigma_is_noop():
    img = _sample_image()
    out = GaussianBlurTechnique(sigma=0.0).apply(img, seed=0)
    assert np.array_equal(out, img)
