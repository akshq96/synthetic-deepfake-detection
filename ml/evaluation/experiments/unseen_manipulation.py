"""Leave-one-manipulation-out generalization: the script that directly
answers the project's primary research question — does synthetic data
augmentation improve generalization to a manipulation technique the model
never saw during training?

For each fake `manipulation_type` present in the manifest, trains a model
with all train-split rows of that type excluded, then evaluates specifically
on val-split rows of that (held-out) type. Comparing this across a
synthetic_ratio=0 run and a synthetic_ratio>0 run (same held-out types) is
how the research question gets an actual answer — this script produces the
per-type numbers; the comparison itself is a downstream analysis (also see
ml/plotting/generalization_tables.py, Phase 5b/plotting).

Usage:
    python -m ml.evaluation.experiments.unseen_manipulation \
        --config ml/configs/cnn_baseline_debug.yaml \
        --output-dir artifacts/experiments/unseen_manipulation/<timestamp>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ml.data_pipeline.leakage_check import check_no_leakage
from ml.data_pipeline.manifest import load_manifest
from ml.data_pipeline.schema import LABEL_FAKE, SPLIT_TRAIN, SPLIT_VAL
from ml.evaluation.evaluate import evaluate_on_manifest, load_calibration_if_exists, load_trained_model
from ml.training.config import load_config
from ml.training.train import run_training


def list_fake_manipulation_types(manifest: pd.DataFrame) -> list[str]:
    fake_rows = manifest[manifest["label"] == LABEL_FAKE]
    types = sorted(t for t in fake_rows["manipulation_type"].dropna().unique() if t)
    return types


def run_unseen_manipulation_experiment(
    cfg,
    *,
    held_out_types: list[str] | None = None,
    eval_split: str = SPLIT_VAL,
) -> dict:
    """Runs one leave-one-manipulation-out training+eval per held-out type.
    Returns {held_out_type: {"train_summary": ..., "eval_metrics": ..., "n_eval_samples": ...}}.
    A type with zero eval-split samples of that type is skipped with a note
    (can happen on small/fixture manifests) rather than raising.
    """
    manifest = load_manifest(cfg.data.manifest_path)
    check_no_leakage(manifest)

    types = held_out_types or list_fake_manipulation_types(manifest)
    if not types:
        raise ValueError(
            f"no fake manipulation_type values found in {cfg.data.manifest_path} — "
            f"nothing to leave out"
        )

    base_run_name = str(cfg.run_name)
    results: dict = {}

    for held_out in types:
        eval_subset = manifest[
            (manifest["split"] == eval_split) & (manifest["manipulation_type"] == held_out)
        ]
        if len(eval_subset) == 0:
            results[held_out] = {"skipped": True, "reason": f"no {eval_split}-split samples of this type"}
            continue

        train_exclude_mask = (manifest["split"] == SPLIT_TRAIN) & (manifest["manipulation_type"] == held_out)
        filtered_manifest = manifest[~train_exclude_mask].reset_index(drop=True)

        run_cfg = cfg.copy()
        run_cfg.run_name = f"{base_run_name}_loo_{held_out}"

        train_summary = run_training(run_cfg, manifest_override=filtered_manifest)

        model = load_trained_model(train_summary["checkpoint_path"], cfg.model)
        calibration = load_calibration_if_exists(train_summary["calibration_path"])
        image_size = int(cfg.data.get("image_size", 224))
        metrics, raw = evaluate_on_manifest(
            model, eval_subset, image_size=image_size, calibration=calibration
        )

        results[held_out] = {
            "skipped": False,
            "held_out_type": held_out,
            "n_eval_samples": len(eval_subset),
            "eval_metrics": metrics.to_dict(),
            "train_summary": train_summary,
        }

    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--held-out-types", nargs="*", default=None, help="Subset of manipulation types to hold out"
    )
    args, overrides = parser.parse_known_args(argv)

    cfg = load_config(args.config, cli_overrides=overrides)
    results = run_unseen_manipulation_experiment(cfg, held_out_types=args.held_out_types)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "unseen_manipulation_results.json"
    results_path.write_text(json.dumps(results, indent=2, default=str))

    print(f"Unseen-manipulation results written to {results_path}")
    for held_out, r in results.items():
        if r.get("skipped"):
            print(f"  {held_out}: skipped ({r['reason']})")
        else:
            print(f"  {held_out}: n={r['n_eval_samples']} metrics={r['eval_metrics']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
