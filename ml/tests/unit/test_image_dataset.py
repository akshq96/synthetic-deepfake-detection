from __future__ import annotations

import pandas as pd
import pytest
import torch

from ml.datasets.image_dataset import ManifestImageDataset
from ml.data_pipeline.schema import empty_manifest


def test_image_dataset_getitem_shapes(fixture_manifest_split: pd.DataFrame):
    train = fixture_manifest_split[fixture_manifest_split["split"] == "train"]
    ds = ManifestImageDataset(train, image_size=32, train=True)
    assert len(ds) == len(train)
    item = ds[0]
    assert item["image"].shape == (3, 32, 32)
    assert item["image"].dtype == torch.float32
    assert item["label"].item() in (0.0, 1.0)
    assert isinstance(item["sample_id"], str)


def test_image_dataset_labels_match_manifest(fixture_manifest_split: pd.DataFrame):
    train = fixture_manifest_split[fixture_manifest_split["split"] == "train"].reset_index(drop=True)
    ds = ManifestImageDataset(train, image_size=16, train=False)
    for i in range(len(ds)):
        item = ds[i]
        expected = 0.0 if train.iloc[i]["label"] == "real" else 1.0
        assert item["label"].item() == expected


def test_image_dataset_rejects_empty_manifest():
    with pytest.raises(ValueError, match="empty manifest"):
        ManifestImageDataset(empty_manifest())
