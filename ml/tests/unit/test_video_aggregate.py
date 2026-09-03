from __future__ import annotations

import pytest

from ml.data_pipeline.schema import LABEL_FAKE, LABEL_REAL
from ml.models.predictor import Prediction
from ml.video.aggregate import FramePrediction, aggregate_track, aggregate_video, top_suspicious_frames


def _fp(track_id: int, frame_index: int, fake_prob: float, *, abstained: bool = False) -> FramePrediction:
    label = "abstain" if abstained else (LABEL_FAKE if fake_prob >= 0.5 else LABEL_REAL)
    confidence = fake_prob if label == LABEL_FAKE else (1 - fake_prob if label == LABEL_REAL else fake_prob)
    pred = Prediction(label=label, confidence=confidence, fake_probability=fake_prob, abstained=abstained)
    return FramePrediction(track_id=track_id, frame_index=frame_index, timestamp=frame_index / 10.0, prediction=pred)


def test_aggregate_track_mean_prob():
    preds = [_fp(0, 0, 0.2), _fp(0, 1, 0.8)]
    result = aggregate_track(preds, method="mean_prob")
    assert result["fake_probability"] == pytest.approx(0.5)
    assert result["n_frames"] == 2
    assert result["n_abstained"] == 0


def test_aggregate_track_max_prob():
    preds = [_fp(0, 0, 0.2), _fp(0, 1, 0.9)]
    result = aggregate_track(preds, method="max_prob")
    assert result["fake_probability"] == pytest.approx(0.9)
    assert result["label"] == LABEL_FAKE


def test_aggregate_track_majority_vote():
    preds = [_fp(0, 0, 0.9), _fp(0, 1, 0.9), _fp(0, 2, 0.1)]
    result = aggregate_track(preds, method="majority_vote")
    assert result["fake_probability"] == pytest.approx(2 / 3)
    assert result["label"] == LABEL_FAKE


def test_aggregate_track_excludes_abstained_frames():
    preds = [_fp(0, 0, 0.9), _fp(0, 1, 0.0, abstained=True)]
    result = aggregate_track(preds, method="mean_prob")
    assert result["fake_probability"] == pytest.approx(0.9)
    assert result["n_frames"] == 2
    assert result["n_abstained"] == 1


def test_aggregate_track_all_abstained_returns_abstain():
    preds = [_fp(0, 0, 0.5, abstained=True), _fp(0, 1, 0.5, abstained=True)]
    result = aggregate_track(preds, method="mean_prob")
    assert result["label"] == "abstain"
    assert result["fake_probability"] is None


def test_aggregate_track_unknown_method_raises():
    with pytest.raises(ValueError, match="unknown aggregation method"):
        aggregate_track([_fp(0, 0, 0.5)], method="not_a_method")


def test_aggregate_video_flags_fake_if_any_track_fake():
    preds = [_fp(0, 0, 0.1), _fp(1, 0, 0.95)]  # track 0 real, track 1 confidently fake
    result = aggregate_video(preds)
    assert result["video_label"] == LABEL_FAKE
    assert result["n_tracks"] == 2


def test_aggregate_video_real_if_all_tracks_real():
    preds = [_fp(0, 0, 0.1), _fp(1, 0, 0.2)]
    result = aggregate_video(preds)
    assert result["video_label"] == LABEL_REAL


def test_aggregate_video_abstain_if_no_predictions():
    result = aggregate_video([])
    assert result["video_label"] == "abstain"
    assert result["n_tracks"] == 0


def test_aggregate_video_respects_fake_threshold():
    preds = [_fp(0, 0, 0.6)]  # fake but only mildly so
    result_low_threshold = aggregate_video(preds, fake_threshold=0.5)
    result_high_threshold = aggregate_video(preds, fake_threshold=0.9)
    assert result_low_threshold["video_label"] == LABEL_FAKE
    assert result_high_threshold["video_label"] == LABEL_REAL


def test_top_suspicious_frames_orders_by_fake_probability_descending():
    preds = [_fp(0, 0, 0.1), _fp(0, 1, 0.9), _fp(0, 2, 0.5)]
    top = top_suspicious_frames(preds, k=2)
    assert [fp.frame_index for fp in top] == [1, 2]


def test_top_suspicious_frames_excludes_abstained():
    preds = [_fp(0, 0, 0.99, abstained=True), _fp(0, 1, 0.7)]
    top = top_suspicious_frames(preds, k=5)
    assert len(top) == 1
    assert top[0].frame_index == 1
