"""Ablation sweep harness: trains+evaluates one run per point in a config
grid (e.g. synthetic_ratio x synthetic_techniques), sequentially — real runs
have exactly one GPU/CPU available, so sweeps are not parallelized here.
Each point is logged as its own MLflow run (tagged with the sweep id) via
the normal run_training path, and the whole sweep's results are written to
one JSON file for ml.plotting.ablation_tables / the backend's ablation
results endpoint to read without recomputation.

Usage:
    python -m ml.evaluation.ablation --config ml/configs/cnn_baseline_debug.yaml \
        --grid '{"data.synthetic_ratio": [0.0, 0.3, 0.5]}' \
        --output-dir artifacts/experiments/ablation/<timestamp>
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

from omegaconf import OmegaConf

from ml.data_pipeline.leakage_check import check_no_leakage
from ml.data_pipeline.manifest import load_manifest
from ml.data_pipeline.schema import SPLIT_VAL
from ml.evaluation.evaluate import evaluate_on_manifest, load_calibration_if_exists, load_trained_model
from ml.training.config import load_config
from ml.training.train import run_training


def _grid_points(grid: dict[str, list]) -> list[dict]:
    """Cartesian product of a {config_key: [values]} grid -> a list of
    {config_key: value} points, e.g. {"a": [1,2], "b": [3,4]} -> 4 points.
    """
    if not grid:
        return [{}]
    keys = list(grid.keys())
    value_lists = [grid[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*value_lists)]


def run_ablation_sweep(
    cfg,
    *,
    grid: dict[str, list],
    sweep_id: str | None = None,
    eval_split: str = SPLIT_VAL,
) -> list[dict]:
    """Runs one training+eval per grid point. Returns a list of
    {point, run_summary, eval_metrics} dicts, in grid-iteration order.
    """
    sweep_id = sweep_id or str(cfg.run_name)
    points = _grid_points(grid)

    manifest = load_manifest(cfg.data.manifest_path)
    check_no_leakage(manifest)
    eval_manifest = manifest[manifest["split"] == eval_split]
    if len(eval_manifest) == 0:
        raise ValueError(f"no {eval_split}-split rows in {cfg.data.manifest_path} to evaluate the sweep on")

    results = []
    for i, point in enumerate(points):
        run_cfg = cfg.copy()
        for dotted_key, value in point.items():
            OmegaConf.update(run_cfg, dotted_key, value, merge=True)
        run_cfg.run_name = f"{sweep_id}_point{i}"

        run_summary = run_training(run_cfg)

        model = load_trained_model(run_summary["checkpoint_path"], run_cfg.model)
        calibration = load_calibration_if_exists(run_summary["calibration_path"])
        image_size = int(run_cfg.data.get("image_size", 224))
        metrics, _ = evaluate_on_manifest(model, eval_manifest, image_size=image_size, calibration=calibration)

        results.append({"point": point, "run_summary": run_summary, "eval_metrics": metrics.to_dict()})

    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--grid", required=True, help="JSON dict of {config.dotted.key: [values]}")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sweep-id", default=None)
    args, overrides = parser.parse_known_args(argv)

    cfg = load_config(args.config, cli_overrides=overrides)
    grid = json.loads(args.grid)

    results = run_ablation_sweep(cfg, grid=grid, sweep_id=args.sweep_id)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "ablation_results.json"
    results_path.write_text(json.dumps(results, indent=2, default=str))

    print(f"Ablation sweep results written to {results_path}")
    for r in results:
        print(f"  {r['point']}: {r['eval_metrics']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
