"""PyTorch Dataset over a split manifest.

Deliberately dumb: given an already-built, already-split, already-leakage-
checked manifest DataFrame, this class just loads images and labels for one
`split`. It does not know about synthetic-data ratios (that mixing logic is
`ml/synthetic/mixer.py`, Phase 3) — it simply loads whatever rows the caller
hands it, so the mixer can pass in a pre-filtered/pre-sampled manifest slice
without this class needing to change.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ml.data_pipeline.schema import LABEL_REAL

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def default_transform(image_size: int, *, train: bool):
    import torchvision.transforms as T

    ops = [T.ToPILImage(), T.Resize((image_size, image_size))]
    if train:
        ops.append(T.RandomHorizontalFlip(p=0.5))
    ops += [T.ToTensor(), T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]
    return T.Compose(ops)


class ManifestImageDataset(Dataset):
    """One sample = one row of `manifest` (already filtered to the desired
    split by the caller). Label: 0.0 = real, 1.0 = fake.
    """

    def __init__(
        self,
        manifest: pd.DataFrame,
        *,
        image_size: int = 224,
        train: bool = False,
        transform=None,
    ):
        if len(manifest) == 0:
            raise ValueError(
                "ManifestImageDataset received an empty manifest slice — check the "
                "split filter / synthetic-ratio sampling upstream."
            )
        self._manifest = manifest.reset_index(drop=True)
        self._transform = transform or default_transform(image_size, train=train)

    def __len__(self) -> int:
        return len(self._manifest)

    def __getitem__(self, idx: int) -> dict:
        import cv2

        row = self._manifest.iloc[idx]
        image_bgr = cv2.imread(row["file_path"])
        if image_bgr is None:
            raise FileNotFoundError(f"could not read image at {row['file_path']!r}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_tensor = self._transform(image_rgb)

        label = 0.0 if row["label"] == LABEL_REAL else 1.0
        return {
            "image": image_tensor,
            "label": torch.tensor(label, dtype=torch.float32),
            "sample_id": row["sample_id"],
            "is_synthetic": bool(row["is_synthetic"]),
        }

    @property
    def manifest(self) -> pd.DataFrame:
        return self._manifest
