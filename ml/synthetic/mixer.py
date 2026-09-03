"""Synthetic-data ratio/technique mixing: the knob the whole research
question (does synthetic augmentation help generalization?) is built around.

Critical constraint enforced here, not just documented: synthetic samples are
generated ONLY from train-split real rows, and each synthetic sample inherits
its source row's `identity_id`, `source_video_id`, and `split` — so it can
never introduce leakage (it's definitionally in the same split as its
source) and never contaminates val/test (synthetic augmentation is a
training-time-only technique; evaluation must reflect the real-data
distribution being generalized to).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ml.data_pipeline.manifest import build_manifest, make_row, merge_manifests
from ml.data_pipeline.schema import LABEL_FAKE, SPLIT_TRAIN
from ml.synthetic.autoencoder_swap import AutoencoderSwapTechnique
from ml.synthetic.base import SyntheticTechnique
from ml.synthetic.blend_warp import BlendWarpTechnique
from ml.synthetic.color_perturb import ColorPerturbTechnique
from ml.synthetic.compression_artifact import CompressionArtifactTechnique
from ml.synthetic.freq_perturb import FreqPerturbTechnique

TECHNIQUE_REGISTRY: dict[str, type[SyntheticTechnique]] = {
    "blend_warp": BlendWarpTechnique,
    "freq_perturb": FreqPerturbTechnique,
    "compression_artifact": CompressionArtifactTechnique,
    "color_perturb": ColorPerturbTechnique,
    "autoencoder_swap": AutoencoderSwapTechnique,
}


def build_technique(name: str) -> SyntheticTechnique:
    if name not in TECHNIQUE_REGISTRY:
        raise ValueError(f"unknown synthetic technique {name!r}; supported: {sorted(TECHNIQUE_REGISTRY)}")
    return TECHNIQUE_REGISTRY[name]()


def n_synthetic_for_ratio(n_real: int, ratio: float) -> int:
    """How many synthetic samples to add to `n_real` real samples so that
    synthetic / (real + synthetic) == ratio. Ratio must be in [0, 1).
    """
    if not (0.0 <= ratio < 1.0):
        raise ValueError(f"synthetic_ratio must be in [0, 1), got {ratio}")
    if ratio == 0.0:
        return 0
    return int(round(ratio * n_real / (1.0 - ratio)))


def generate_synthetic_samples(
    base_manifest: pd.DataFrame,
    *,
    techniques: list[str],
    ratio: float,
    output_dir: str | Path,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic manipulated samples from the manifest's train-split
    real rows, cycling through `techniques`. Returns a new manifest (not
    merged with `base_manifest` — see `mix_into_manifest`).
    """
    import cv2

    if not techniques:
        raise ValueError("techniques must be a non-empty list when ratio > 0")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_pool = base_manifest[
        (base_manifest["split"] == SPLIT_TRAIN) & (~base_manifest["is_synthetic"])
    ].reset_index(drop=True)
    if len(source_pool) == 0:
        raise ValueError("no train-split real rows available to synthesize from")

    n_synth = n_synthetic_for_ratio(len(source_pool), ratio)
    rng = np.random.default_rng(seed)
    if n_synth <= len(source_pool):
        source_indices = rng.choice(len(source_pool), size=n_synth, replace=False)
    else:
        source_indices = rng.choice(len(source_pool), size=n_synth, replace=True)

    technique_instances = {name: build_technique(name) for name in techniques}
    rows = []
    for i, src_idx in enumerate(source_indices):
        src_row = source_pool.iloc[int(src_idx)]
        technique_name = techniques[i % len(techniques)]
        technique = technique_instances[technique_name]

        image = cv2.imread(src_row["file_path"])
        if image is None:
            raise FileNotFoundError(f"could not read source image {src_row['file_path']!r}")

        per_sample_seed = int(seed) * 1_000_003 + i
        synthetic_image = technique.apply(image, seed=per_sample_seed)

        out_name = f"{technique_name}_{src_row['sample_id']}_{i}.jpg"
        out_path = output_dir / out_name
        cv2.imwrite(str(out_path), synthetic_image)

        rows.append(
            make_row(
                source_dataset=src_row["source_dataset"],
                source_video_id=src_row["source_video_id"],
                identity_id=src_row["identity_id"],
                file_path=str(out_path),
                label=LABEL_FAKE,  # synthetic manipulations are, by construction, fake samples
                manipulation_type=f"synthetic_{technique_name}",
                is_synthetic=True,
                synthetic_technique=technique_name,
                quality_ok=True,
            )
        )

    synthetic_manifest = build_manifest(rows)
    # Inherit split from source rows (all SPLIT_TRAIN here, by construction of source_pool).
    synthetic_manifest["split"] = SPLIT_TRAIN
    return synthetic_manifest


def mix_into_manifest(
    base_manifest: pd.DataFrame,
    *,
    ratio: float,
    techniques: list[str],
    output_dir: str | Path,
    seed: int = 42,
) -> pd.DataFrame:
    """Return `base_manifest` with synthetic train-split samples added per
    `ratio`/`techniques`. If ratio == 0, returns `base_manifest` unchanged
    (no-op — this is the "baseline, no synthetic augmentation" condition the
    baseline-vs-synthetic experiment compares against).
    """
    if ratio == 0.0:
        return base_manifest
    synthetic_manifest = generate_synthetic_samples(
        base_manifest, techniques=techniques, ratio=ratio, output_dir=output_dir, seed=seed
    )
    return merge_manifests([base_manifest, synthetic_manifest])
