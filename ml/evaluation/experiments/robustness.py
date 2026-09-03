"""Robustness evaluation: apply perturbations (Gaussian noise, Gaussian
blur, JPEG compression) at increasing severity to a fixed evaluation set and
measure accuracy/ROC-AUC degradation — producing the robustness curves the
project plan calls for.

Methodological caveat (also in ml/synthetic/compression_artifact.py and
docs/methodology.md once written): compression is also one of the five
synthetic training techniques. A checkpoint trained with `compression_
artifact` in its synthetic mix should not have its "robustness to
compression" reported without flagging that the result is expected/
controlled (the model was shown compression artifacts during training) —
`compression_in_training_mix` is recorded in this script's results for
exactly that reason, so downstream reporting can't silently conflate the two.

Usage:
    python -m ml.evaluation.experiments.robustness \
        --config ml/configs/cnn_baseline_debug.yaml \
        --checkpoint artifacts/checkpoints/<run>/best.pt \
        --output-dir artifacts/experiments/robustness/<timestamp>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import pandas as pd

from ml.data_pipeline.manifest import build_manifest, load_manifest, make_row
from ml.evaluation.evaluate import evaluate_on_manifest, load_calibration_if_exists, load_trained_model
from ml.synthetic.base import SyntheticTechnique
from ml.synthetic.compression_artifact import CompressionArtifactTechnique
from ml.synthetic.gaussian_blur import GaussianBlurTechnique
from ml.synthetic.gaussian_noise import GaussianNoiseTechnique
from ml.training.config import load_config

DEFAULT_SEVERITIES: dict[str, list[float]] = {
    "gaussian_noise": [5.0, 15.0, 30.0, 50.0],
    "gaussian_blur": [0.5, 1.5, 3.0, 5.0],
    "compression": [80, 50, 30, 15],  # JPEG quality — lower quality = more severe
}


def _build_perturbation(name: str, severity: float) -> SyntheticTechnique:
    if name == "gaussian_noise":
        return GaussianNoiseTechnique(sigma=severity)
    if name == "gaussian_blur":
        return GaussianBlurTechnique(sigma=severity)
    if name == "compression":
        quality = int(severity)
        return CompressionArtifactTechnique(quality_range=(quality, quality), double_compress_prob=0.0)
    raise ValueError(f"unknown robustness perturbation {name!r}; expected one of {list(DEFAULT_SEVERITIES)}")


def perturb_manifest(
    manifest: pd.DataFrame, technique: SyntheticTechnique, *, output_dir: str | Path, seed: int = 42
) -> pd.DataFrame:
    """Apply `technique` to every image in `manifest`, writing perturbed
    copies and returning a new manifest with the SAME labels/identities/split
    (unlike ml.synthetic.mixer, which manufactures new fake samples — a
    robustness perturbation doesn't change ground truth, it tests whether the
    model's prediction is stable under it).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, (_, row) in enumerate(manifest.iterrows()):
        image = cv2.imread(row["file_path"])
        if image is None:
            raise FileNotFoundError(f"could not read {row['file_path']!r}")
        perturbed = technique.apply(image, seed=seed * 1_000_003 + i)

        out_path = output_dir / f"{row['sample_id']}.jpg"
        cv2.imwrite(str(out_path), perturbed)

        rows.append(
            make_row(
                source_dataset=row["source_dataset"],
                source_video_id=row["source_video_id"],
                identity_id=row["identity_id"],
                file_path=str(out_path),
                label=row["label"],
                manipulation_type=row["manipulation_type"],
                is_synthetic=bool(row["is_synthetic"]),
                synthetic_technique=row["synthetic_technique"] if pd.notna(row["synthetic_technique"]) else None,
                quality_ok=True,
                sample_id=f"{row['sample_id']}_{technique.name}",
            )
        )
    out_manifest = build_manifest(rows)
    out_manifest["split"] = manifest["split"].values
    return out_manifest


def run_robustness_experiment(
    cfg,
    *,
    checkpoint_path: str,
    calibration_path: str,
    eval_manifest: pd.DataFrame,
    output_dir: str | Path,
    perturbations: dict[str, list[float]] | None = None,
    compression_in_training_mix: bool = False,
    seed: int = 42,
) -> dict:
    perturbations = perturbations if perturbations is not None else DEFAULT_SEVERITIES
    output_dir = Path(output_dir)

    model = load_trained_model(checkpoint_path, cfg.model)
    calibration = load_calibration_if_exists(calibration_path)
    image_size = int(cfg.data.get("image_size", 224))

    baseline_metrics, _ = evaluate_on_manifest(model, eval_manifest, image_size=image_size, calibration=calibration)

    results: dict = {
        "checkpoint_path": str(checkpoint_path),
        "compression_in_training_mix": compression_in_training_mix,
        "baseline_metrics": baseline_metrics.to_dict(),
        "perturbations": {},
    }

    for pert_name, severities in perturbations.items():
        curve = []
        for severity in severities:
            technique = _build_perturbation(pert_name, severity)
            perturbed_manifest = perturb_manifest(
                eval_manifest, technique, output_dir=output_dir / pert_name / str(severity), seed=seed
            )
            metrics, _ = evaluate_on_manifest(
                model, perturbed_manifest, image_size=image_size, calibration=calibration
            )
            curve.append({"severity": severity, "metrics": metrics.to_dict()})
        results["perturbations"][pert_name] = curve

    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--calibration", default=None, help="Defaults to <checkpoint_dir>/temperature.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--compression-in-training-mix",
        action="store_true",
        help="Set if this checkpoint's synthetic_techniques included compression_artifact",
    )
    args, overrides = parser.parse_known_args(argv)

    cfg = load_config(args.config, cli_overrides=overrides)
    manifest = load_manifest(cfg.data.manifest_path)
    from ml.data_pipeline.schema import SPLIT_TEST

    eval_manifest = manifest[manifest["split"] == SPLIT_TEST]
    if len(eval_manifest) == 0:
        eval_manifest = manifest[manifest["split"] == "val"]

    calibration_path = args.calibration or str(Path(args.checkpoint).parent / "temperature.json")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = run_robustness_experiment(
        cfg,
        checkpoint_path=args.checkpoint,
        calibration_path=calibration_path,
        eval_manifest=eval_manifest,
        output_dir=output_dir,
        compression_in_training_mix=args.compression_in_training_mix,
    )

    results_path = output_dir / "robustness_results.json"
    results_path.write_text(json.dumps(results, indent=2, default=str))

    from ml.plotting.robustness_curves import plot_robustness_curve

    plot_path = plot_robustness_curve(results, output_dir / "robustness_accuracy.png")

    print(f"Robustness results written to {results_path}")
    print(f"Robustness plot written to {plot_path}")
    print(f"  baseline: {results['baseline_metrics']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
