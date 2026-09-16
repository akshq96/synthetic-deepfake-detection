"""Real-data answer to the project's research question on FaceForensics++ C32.

For each held-out manipulation type H and each synthetic ratio r in {0, 0.3}:
  1. remove every train-split row of type H,
  2. train EfficientNetV2-S with the framework's own run_training()
     (leakage check, synthetic mixing via ml.synthetic.mixer, temperature scaling),
  3. evaluate on the TEST split restricted to real rows + rows of type H
     (a type the model never saw), reporting AUC, accuracy, precision, recall, F1.

Comparing r=0 against r=0.3 for the same H is the augmentation effect.

    python -m ml.evaluation.experiments.ffpp_augmentation_study \
        --manifest data/manifests/ffpp_c32_split.parquet \
        --held-out FaceShifter NeuralTextures --epochs 4 --image-size 112 \
        --output artifacts/experiments/ffpp_augmentation/results.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf

from ml.data_pipeline.leakage_check import check_no_leakage
from ml.data_pipeline.manifest import load_manifest
from ml.data_pipeline.schema import SPLIT_TEST, SPLIT_TRAIN
from ml.evaluation.evaluate import evaluate_on_manifest, load_calibration_if_exists, load_trained_model
from ml.training.train import run_training

TECHNIQUES = ["blend_warp", "freq_perturb", "compression_artifact", "color_perturb", "autoencoder_swap"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--held-out", nargs="+", default=["FaceShifter", "NeuralTextures"])
    ap.add_argument("--ratios", nargs="+", type=float, default=[0.0, 0.3])
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--image-size", type=int, default=112)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", required=True)
    args = ap.parse_args(argv)

    manifest = load_manifest(args.manifest)
    check_no_leakage(manifest)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = json.loads(out_path.read_text()) if out_path.exists() else {}

    for held in args.held_out:
        train_drop = (manifest["split"] == SPLIT_TRAIN) & (manifest["manipulation_type"] == held)
        filtered = manifest[~train_drop].reset_index(drop=True)
        test_rows = manifest[(manifest["split"] == SPLIT_TEST) &
                             ((manifest["label"] == "real") | (manifest["manipulation_type"] == held))]
        for ratio in args.ratios:
            key = f"{held}|ratio={ratio}"
            if key in results:
                print("skip (done):", key)
                continue
            cfg = OmegaConf.create({
                "experiment_name": "ffpp_augmentation",
                "run_name": f"ffpp_loo_{held}_r{int(ratio * 100)}",
                "artifacts_root": "artifacts",
                "mlflow_tracking_uri": None,
                "model": {"name": "efficientnetv2_s", "pretrained": False, "drop_rate": 0.1},
                "data": {"manifest_path": args.manifest, "image_size": args.image_size,
                         "synthetic_ratio": ratio, "synthetic_techniques": TECHNIQUES if ratio > 0 else []},
                "training": {"debug": False, "epochs": args.epochs, "batch_size": args.batch_size,
                             "lr": args.lr, "seed": args.seed, "resume": None},
            })
            t0 = time.time()
            summary = run_training(cfg, manifest_override=filtered)
            model = load_trained_model(summary["best_checkpoint_path"] or summary["checkpoint_path"], cfg.model, device=device)
            cal = load_calibration_if_exists(summary["calibration_path"])
            metrics, _ = evaluate_on_manifest(model, test_rows, image_size=args.image_size, calibration=cal, device=device)
            results[key] = {
                "held_out_type": held, "synthetic_ratio": ratio,
                "n_train": int(((filtered["split"] == SPLIT_TRAIN)).sum()),
                "n_test": len(test_rows), "n_test_fake_heldout": int((test_rows["label"] == "fake").sum()),
                "test_metrics": metrics.to_dict(), "val_metrics": summary["val_metrics"],
                "minutes": round((time.time() - t0) / 60, 1),
            }
            out_path.write_text(json.dumps(results, indent=2, default=str))
            print(key, json.dumps(results[key]["test_metrics"]), f"{results[key]['minutes']} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
