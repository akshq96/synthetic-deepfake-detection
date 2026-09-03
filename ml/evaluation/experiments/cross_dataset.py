"""Cross-dataset generalization: train on one `source_dataset`'s train split,
evaluate on a different dataset's test split (and optionally the reverse
direction). Reports the generalization drop, i.e. how much worse the model
does on data it never saw during training or even dataset curation, which is
a stronger generalization test than a held-out split of the *same* dataset.

Usage:
    python -m ml.evaluation.experiments.cross_dataset \
        --config ml/configs/cnn_baseline_debug.yaml \
        --train-dataset ffpp --eval-dataset celebdf \
        --output-dir artifacts/experiments/cross_dataset/<timestamp>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ml.data_pipeline.leakage_check import check_no_leakage
from ml.data_pipeline.manifest import load_manifest
from ml.data_pipeline.schema import SPLIT_TEST, SPLIT_TRAIN
from ml.evaluation.evaluate import evaluate_on_manifest, load_calibration_if_exists, load_trained_model
from ml.training.config import load_config
from ml.training.train import run_training


def run_cross_dataset_experiment(
    cfg,
    *,
    train_dataset: str,
    eval_dataset: str,
    eval_split: str = SPLIT_TEST,
) -> dict:
    """Trains on `train_dataset`'s train split only, evaluates on
    `eval_dataset`'s `eval_split`. Also reports same-dataset (in-distribution)
    metrics on `train_dataset`'s own `eval_split`, as the reference point the
    cross-dataset drop is measured against.
    """
    manifest = load_manifest(cfg.data.manifest_path)
    check_no_leakage(manifest)

    available_datasets = sorted(manifest["source_dataset"].unique())
    for name in (train_dataset, eval_dataset):
        if name not in available_datasets:
            raise ValueError(f"dataset {name!r} not found in manifest; available: {available_datasets}")

    train_only_manifest = manifest[manifest["source_dataset"] == train_dataset].reset_index(drop=True)
    if (train_only_manifest["split"] == SPLIT_TRAIN).sum() == 0:
        raise ValueError(f"no train-split rows for dataset {train_dataset!r}")

    run_cfg = cfg.copy()
    run_cfg.run_name = f"{cfg.run_name}_train_{train_dataset}"
    train_summary = run_training(run_cfg, manifest_override=train_only_manifest)

    model = load_trained_model(train_summary["checkpoint_path"], cfg.model)
    calibration = load_calibration_if_exists(train_summary["calibration_path"])
    image_size = int(cfg.data.get("image_size", 224))

    in_distribution_subset = manifest[
        (manifest["source_dataset"] == train_dataset) & (manifest["split"] == eval_split)
    ]
    cross_dataset_subset = manifest[
        (manifest["source_dataset"] == eval_dataset) & (manifest["split"] == eval_split)
    ]

    result: dict = {
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "train_summary": train_summary,
    }

    if len(in_distribution_subset) > 0:
        in_dist_metrics, _ = evaluate_on_manifest(
            model, in_distribution_subset, image_size=image_size, calibration=calibration
        )
        result["in_distribution_metrics"] = in_dist_metrics.to_dict()
    else:
        result["in_distribution_metrics"] = None

    if len(cross_dataset_subset) > 0:
        cross_metrics, _ = evaluate_on_manifest(
            model, cross_dataset_subset, image_size=image_size, calibration=calibration
        )
        result["cross_dataset_metrics"] = cross_metrics.to_dict()
        if result["in_distribution_metrics"]:
            result["generalization_gap_accuracy"] = (
                result["in_distribution_metrics"]["accuracy"] - cross_metrics.to_dict()["accuracy"]
            )
    else:
        result["cross_dataset_metrics"] = None
        result["generalization_gap_accuracy"] = None

    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--train-dataset", required=True)
    parser.add_argument("--eval-dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    args, overrides = parser.parse_known_args(argv)

    cfg = load_config(args.config, cli_overrides=overrides)
    result = run_cross_dataset_experiment(
        cfg, train_dataset=args.train_dataset, eval_dataset=args.eval_dataset
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "cross_dataset_results.json"
    results_path.write_text(json.dumps(result, indent=2, default=str))

    print(f"Cross-dataset results written to {results_path}")
    print(f"  in-distribution: {result['in_distribution_metrics']}")
    print(f"  cross-dataset:   {result['cross_dataset_metrics']}")
    print(f"  generalization gap (accuracy): {result.get('generalization_gap_accuracy')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
