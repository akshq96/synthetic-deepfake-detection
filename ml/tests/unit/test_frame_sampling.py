from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ml.data_pipeline.frame_sampling import (
    extract_frames,
    sample_frame_indices,
    video_frame_count,
)


def test_sample_frame_indices_uniform_within_bounds():
    idxs = sample_frame_indices(100, 10, strategy="uniform")
    assert len(idxs) == 10
    assert idxs[0] == 0
    assert idxs[-1] == 99
    assert idxs == sorted(idxs)
    assert len(set(idxs)) == len(idxs)


def test_sample_frame_indices_short_video_returns_all_frames():
    idxs = sample_frame_indices(5, 32, strategy="uniform")
    assert idxs == [0, 1, 2, 3, 4]


def test_sample_frame_indices_zero_frames():
    assert sample_frame_indices(0, 10) == []


def test_sample_frame_indices_every_nth():
    idxs = sample_frame_indices(100, 10, strategy="every_nth")
    assert len(idxs) <= 10
    assert idxs == sorted(idxs)


def test_sample_frame_indices_unknown_strategy_raises():
    with pytest.raises(ValueError, match="unknown sampling strategy"):
        sample_frame_indices(10, 5, strategy="bogus")


def _write_tiny_video(path: Path, n_frames: int = 10, size: int = 32) -> None:
    import cv2

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, 10.0, (size, size))
    assert writer.isOpened(), "test environment cannot write video via OpenCV MJPG codec"
    rng = np.random.default_rng(0)
    for i in range(n_frames):
        frame = np.full((size, size, 3), fill_value=(i * 20) % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_video_frame_count_and_extract_frames(tmp_path: Path):
    video_path = tmp_path / "tiny.avi"
    _write_tiny_video(video_path, n_frames=10)

    count = video_frame_count(str(video_path))
    assert count == 10

    idxs = sample_frame_indices(count, 4)
    extracted = list(extract_frames(str(video_path), idxs))
    assert [i for i, _frame in extracted] == sorted(idxs)
    for _idx, frame in extracted:
        assert frame.shape == (32, 32, 3)


def test_extract_frames_missing_video_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        list(extract_frames(str(tmp_path / "nope.avi"), [0, 1]))
