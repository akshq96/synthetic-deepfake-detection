from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from ml.training.checkpoint import (
    checkpoint_dir,
    find_latest_checkpoint,
    load_checkpoint,
    save_checkpoint,
)


def _tiny_model_and_optimizer():
    model = nn.Linear(4, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    return model, optimizer


def test_save_and_load_checkpoint_roundtrip(tmp_path: Path):
    model, optimizer = _tiny_model_and_optimizer()
    path = tmp_path / "ckpt" / "latest.pt"
    save_checkpoint(path, model=model, optimizer=optimizer, epoch=3, global_step=100, best_metric=0.75)

    assert path.exists()
    state = load_checkpoint(path)
    assert state.epoch == 3
    assert state.global_step == 100
    assert state.best_metric == 0.75

    new_model, new_optimizer = _tiny_model_and_optimizer()
    new_model.load_state_dict(state.model_state)
    new_optimizer.load_state_dict(state.optimizer_state)
    for p1, p2 in zip(model.parameters(), new_model.parameters()):
        assert torch.allclose(p1, p2)


def test_load_checkpoint_missing_raises(tmp_path: Path):
    import pytest

    with pytest.raises(FileNotFoundError, match="checkpoint not found"):
        load_checkpoint(tmp_path / "nope.pt")


def test_find_latest_checkpoint(tmp_path: Path):
    run_id = "my_run"
    assert find_latest_checkpoint(tmp_path, run_id) is None

    model, optimizer = _tiny_model_and_optimizer()
    latest_path = checkpoint_dir(tmp_path, run_id) / "latest.pt"
    save_checkpoint(latest_path, model=model, optimizer=optimizer, epoch=0, global_step=1, best_metric=0.0)

    found = find_latest_checkpoint(tmp_path, run_id)
    assert found == latest_path
