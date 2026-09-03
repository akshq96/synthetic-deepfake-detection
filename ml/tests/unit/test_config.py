from __future__ import annotations

from pathlib import Path

import pytest

from ml.training.config import flatten_for_logging, load_config


def test_load_config_reads_yaml(tmp_path: Path):
    path = tmp_path / "cfg.yaml"
    path.write_text("model:\n  name: efficientnetv2_s\ntraining:\n  lr: 0.001\n")
    cfg = load_config(path)
    assert cfg.model.name == "efficientnetv2_s"
    assert cfg.training.lr == 0.001


def test_load_config_applies_cli_overrides(tmp_path: Path):
    path = tmp_path / "cfg.yaml"
    path.write_text("training:\n  lr: 0.001\n  epochs: 1\n")
    cfg = load_config(path, cli_overrides=["training.lr=0.01"])
    assert cfg.training.lr == 0.01
    assert cfg.training.epochs == 1


def test_load_config_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="config not found"):
        load_config(tmp_path / "nope.yaml")


def test_flatten_for_logging():
    from omegaconf import OmegaConf

    cfg = OmegaConf.create(
        {"model": {"name": "vit", "pretrained": True}, "training": {"lr": 0.001}}
    )
    flat = flatten_for_logging(cfg)
    assert flat == {"model.name": "vit", "model.pretrained": True, "training.lr": 0.001}


def test_flatten_for_logging_handles_lists():
    from omegaconf import OmegaConf

    cfg = OmegaConf.create({"data": {"synthetic_techniques": ["blend_warp", "freq_perturb"]}})
    flat = flatten_for_logging(cfg)
    assert flat["data.synthetic_techniques"] == "blend_warp,freq_perturb"
