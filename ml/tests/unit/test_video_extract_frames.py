from __future__ import annotations

from pathlib import Path

import pytest

from ml.tests.fixtures.synthetic_fixture import generate_fixture_video
from ml.video.extract_frames import (
    frame_index_to_timestamp,
    sample_inference_frame_indices,
    sample_inference_frames,
    video_fps,
)


def test_sample_inference_frame_indices_basic():
    idxs = sample_inference_frame_indices(100, frame_interval=15, max_frames=64)
    assert idxs == [0, 15, 30, 45, 60, 75, 90]


def test_sample_inference_frame_indices_respects_max_frames():
    idxs = sample_inference_frame_indices(1000, frame_interval=1, max_frames=10)
    assert len(idxs) == 10
    assert idxs == list(range(10))


def test_sample_inference_frame_indices_zero_frames():
    assert sample_inference_frame_indices(0) == []


def test_frame_index_to_timestamp():
    assert frame_index_to_timestamp(30, fps=15.0) == 2.0


def test_frame_index_to_timestamp_rejects_non_positive_fps():
    with pytest.raises(ValueError, match="fps must be positive"):
        frame_index_to_timestamp(1, fps=0.0)


def test_video_fps_and_sample_inference_frames(tmp_path: Path):
    video_path = generate_fixture_video(tmp_path / "vid.avi", n_frames=30, fps=10.0)
    fps = video_fps(str(video_path))
    assert fps == pytest.approx(10.0, abs=0.5)

    frames = sample_inference_frames(str(video_path), frame_interval=5, max_frames=10)
    assert len(frames) > 0
    indices = [i for i, _ in frames]
    assert indices == sorted(indices)
    for _idx, frame in frames:
        assert frame.shape == (64, 64, 3)


def test_video_fps_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        video_fps(str(tmp_path / "nope.avi"))
