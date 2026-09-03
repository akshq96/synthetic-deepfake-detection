"""Per-frame predictions -> track-level and video-level verdicts, plus the
"suspicious frames" list the frontend's video-review view needs.

A video is flagged fake if ANY tracked face is predicted fake with high
confidence (per the project plan) — a video can contain one manipulated face
among several real ones, so requiring *all* tracks to agree would miss that.
Frames the model abstained on are excluded from aggregation; if every frame
of a track abstained, the track (and, if it's the only track, the video) is
itself reported as "abstain" rather than forced to a label.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from ml.data_pipeline.schema import LABEL_FAKE, LABEL_REAL
from ml.models.predictor import Prediction

VALID_METHODS = ("mean_prob", "max_prob", "majority_vote")


@dataclass
class FramePrediction:
    track_id: int
    frame_index: int
    timestamp: float
    prediction: Prediction
    heatmap_path: str | None = None


def aggregate_track(frame_predictions: list[FramePrediction], *, method: str = "mean_prob") -> dict:
    if method not in VALID_METHODS:
        raise ValueError(f"unknown aggregation method {method!r}; expected one of {VALID_METHODS}")

    non_abstained = [fp for fp in frame_predictions if not fp.prediction.abstained]
    n_abstained = len(frame_predictions) - len(non_abstained)

    if not non_abstained:
        return {
            "label": "abstain",
            "fake_probability": None,
            "n_frames": len(frame_predictions),
            "n_abstained": n_abstained,
        }

    probs = np.array([fp.prediction.fake_probability for fp in non_abstained])
    if method == "mean_prob":
        agg_prob = float(probs.mean())
    elif method == "max_prob":
        agg_prob = float(probs.max())
    else:  # majority_vote
        agg_prob = float((probs >= 0.5).mean())  # fraction of frames voting "fake"

    label = LABEL_FAKE if agg_prob >= 0.5 else LABEL_REAL
    return {
        "label": label,
        "fake_probability": agg_prob,
        "n_frames": len(frame_predictions),
        "n_abstained": n_abstained,
    }


def aggregate_video(
    frame_predictions: list[FramePrediction], *, method: str = "mean_prob", fake_threshold: float = 0.5
) -> dict:
    by_track: dict[int, list[FramePrediction]] = defaultdict(list)
    for fp in frame_predictions:
        by_track[fp.track_id].append(fp)

    track_results = {track_id: aggregate_track(fps, method=method) for track_id, fps in by_track.items()}

    if not track_results:
        video_label = "abstain"
    elif all(r["label"] == "abstain" for r in track_results.values()):
        video_label = "abstain"
    elif any(
        r["label"] == LABEL_FAKE and (r["fake_probability"] or 0.0) >= fake_threshold
        for r in track_results.values()
    ):
        video_label = LABEL_FAKE
    else:
        video_label = LABEL_REAL

    return {"video_label": video_label, "track_results": track_results, "n_tracks": len(track_results)}


def top_suspicious_frames(frame_predictions: list[FramePrediction], *, k: int = 5) -> list[FramePrediction]:
    """Top-K frames by fake-probability, descending. Abstained frames are
    excluded (an abstained frame is, by definition, not confidently
    suspicious in either direction).
    """
    scored = [fp for fp in frame_predictions if not fp.prediction.abstained]
    scored.sort(key=lambda fp: fp.prediction.fake_probability, reverse=True)
    return scored[:k]
