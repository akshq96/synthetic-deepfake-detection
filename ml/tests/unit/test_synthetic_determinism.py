"""Determinism contract for every synthetic technique: same seed + same
input -> byte-identical output. This is what makes a synthetic-augmentation
ablation run exactly reproducible from its config + seed alone.
"""

from __future__ import annotations

import numpy as np
import pytest

from ml.synthetic.autoencoder_swap import AutoencoderSwapTechnique
from ml.synthetic.blend_warp import BlendWarpTechnique
from ml.synthetic.color_perturb import ColorPerturbTechnique
from ml.synthetic.compression_artifact import CompressionArtifactTechnique
from ml.synthetic.freq_perturb import FreqPerturbTechnique

ALL_TECHNIQUES = [
    BlendWarpTechnique(),
    FreqPerturbTechnique(),
    CompressionArtifactTechnique(),
    ColorPerturbTechnique(),
    AutoencoderSwapTechnique(train_steps=5),  # few steps: fast test, still exercises training
]


def _sample_image(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(48, 48, 3), dtype=np.uint8)


@pytest.mark.parametrize("technique", ALL_TECHNIQUES, ids=lambda t: t.name)
def test_technique_is_deterministic_given_seed(technique):
    img = _sample_image()
    out1 = technique.apply(img, seed=42)
    out2 = technique.apply(img, seed=42)
    assert np.array_equal(out1, out2), f"{technique.name} is not deterministic for a fixed seed"


@pytest.mark.parametrize("technique", ALL_TECHNIQUES, ids=lambda t: t.name)
def test_technique_output_shape_and_dtype(technique):
    img = _sample_image()
    out = technique.apply(img, seed=1)
    assert out.shape == img.shape
    assert out.dtype == np.uint8


@pytest.mark.parametrize("technique", ALL_TECHNIQUES, ids=lambda t: t.name)
def test_technique_changes_the_image(technique):
    img = _sample_image()
    out = technique.apply(img, seed=1)
    assert not np.array_equal(out, img), f"{technique.name} returned the input unchanged"


@pytest.mark.parametrize("technique", ALL_TECHNIQUES, ids=lambda t: t.name)
def test_technique_different_seeds_differ(technique):
    img = _sample_image()
    out1 = technique.apply(img, seed=1)
    out2 = technique.apply(img, seed=2)
    assert not np.array_equal(out1, out2), f"{technique.name} ignores the seed"


def test_technique_does_not_mutate_input():
    img = _sample_image()
    img_copy = img.copy()
    BlendWarpTechnique().apply(img, seed=1)
    assert np.array_equal(img, img_copy)


def test_blend_warp_accepts_donor_image():
    img = _sample_image(seed=0)
    donor = _sample_image(seed=1)
    out = BlendWarpTechnique().apply(img, donor_img=donor, seed=1)
    assert out.shape == img.shape
