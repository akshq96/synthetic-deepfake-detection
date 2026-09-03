"""Phase 8 smoke test: running the full video pipeline on one short fixture
video produces a video-level verdict + suspicious-frames list with heatmap
paths — the definition of done for Phase 8 per the project plan.
"""

from __future__ import annotations

from pathlib import Path

from ml.data_pipeline.face_detector import FaceDetection, MockFaceDetector
from ml.data_pipeline.manifest import save_manifest
from ml.data_pipeline.schema import LABEL_FAKE, LABEL_REAL
from ml.data_pipeline.split import assign_splits
from ml.evaluation.evaluate import load_calibration_if_exists, load_trained_model
from ml.tests.fixtures.synthetic_fixture import generate_fixture_dataset, generate_fixture_video
from ml.training.config import load_config
from ml.training.train import run_training
from ml.video.pipeline import analyze_video
from ml.xai.factory import get_explainer


def test_video_pipeline_smoke(tmp_path: Path):
    raw = generate_fixture_dataset(tmp_path / "raw")
    split = assign_splits(raw, seed=42)
    manifest_path = tmp_path / "manifests" / "fixture_split.parquet"
    save_manifest(split, manifest_path)

    artifacts_root = tmp_path / "artifacts"
    cfg = load_config(
        Path("ml/configs/cnn_baseline_debug.yaml"),
        cli_overrides=[
            f"data.manifest_path={manifest_path}",
            f"artifacts_root={artifacts_root}",
            "run_name=video_pipeline_smoke",
        ],
    )
    train_summary = run_training(cfg)

    model = load_trained_model(train_summary["checkpoint_path"], cfg.model)
    calibration = load_calibration_if_exists(train_summary["calibration_path"])
    explainer = get_explainer(cfg.model.name)

    video_path = generate_fixture_video(tmp_path / "video.avi", n_frames=20, size=64, fps=10.0)
    face_detector = MockFaceDetector(
        detections=[FaceDetection(x1=8, y1=8, x2=56, y2=56, confidence=0.95)]
    )

    result = analyze_video(
        str(video_path),
        model,
        face_detector=face_detector,
        calibration=calibration,
        explainer=explainer,
        heatmap_output_dir=tmp_path / "heatmaps",
        image_size=64,
        frame_interval=5,
        max_frames=8,
        top_k_suspicious=3,
    )

    assert result.video_label in (LABEL_REAL, LABEL_FAKE, "abstain")
    assert result.n_tracks == 1  # MockFaceDetector returns the same bbox every frame
    assert result.n_frames_sampled > 0
    assert 0 < len(result.suspicious_frames) <= 3

    for frame in result.suspicious_frames:
        assert frame["heatmap_path"] is not None
        assert Path(frame["heatmap_path"]).exists()
        assert Path(frame["heatmap_path"]).stat().st_size > 0


def test_video_pipeline_without_explainer_skips_heatmaps(tmp_path: Path):
    raw = generate_fixture_dataset(tmp_path / "raw")
    split = assign_splits(raw, seed=42)
    manifest_path = tmp_path / "manifests" / "fixture_split.parquet"
    save_manifest(split, manifest_path)

    artifacts_root = tmp_path / "artifacts"
    cfg = load_config(
        Path("ml/configs/cnn_baseline_debug.yaml"),
        cli_overrides=[
            f"data.manifest_path={manifest_path}",
            f"artifacts_root={artifacts_root}",
            "run_name=video_pipeline_no_xai_smoke",
        ],
    )
    train_summary = run_training(cfg)
    model = load_trained_model(train_summary["checkpoint_path"], cfg.model)

    video_path = generate_fixture_video(tmp_path / "video2.avi", n_frames=10, size=64)
    face_detector = MockFaceDetector(
        detections=[FaceDetection(x1=8, y1=8, x2=56, y2=56, confidence=0.95)]
    )

    result = analyze_video(
        str(video_path),
        model,
        face_detector=face_detector,
        image_size=64,
        frame_interval=3,
        max_frames=5,
    )
    assert all(frame["heatmap_path"] is None for frame in result.suspicious_frames)
