"""Shared training entrypoint for every architecture (CNN or ViT — purely a
config choice, see ml/models/factory.py). Runs identically on CPU (debug
subset, fast iteration) or Colab/Kaggle GPU (full run) via the same command:

    python -m ml.training.train --config ml/configs/cnn_baseline_debug.yaml
    python -m ml.training.train --config ml/configs/cnn_baseline_debug.yaml \
        training.debug=false training.epochs=20

Before any run starts, the manifest is passed through the leakage checker
(ml.data_pipeline.leakage_check) — this is a hard gate, not a warning.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import expit
from torch import nn
from torch.utils.data import DataLoader

from ml.data_pipeline.leakage_check import check_no_leakage
from ml.data_pipeline.manifest import load_manifest
from ml.data_pipeline.schema import SPLIT_TEST, SPLIT_TRAIN, SPLIT_VAL
from ml.datasets.image_dataset import ManifestImageDataset
from ml.evaluation.metrics import compute_metrics
from ml.models.factory import build_model
from ml.models.predictor import sweep_abstain_margin
from ml.training.calibrate import fit_temperature
from ml.training.checkpoint import checkpoint_dir, find_latest_checkpoint, load_checkpoint, save_checkpoint
from ml.training.config import flatten_for_logging, load_config, REPO_ROOT


def _set_experiment(experiment_name: str, artifacts_root: Path) -> None:
    """mlflow.set_experiment(name) alone would create a new experiment (on
    first use) with mlflow's own default artifact location. We want run
    artifacts to land under our own artifacts/mlflow/<experiment>/ tree
    regardless of the tracking-store backend (sqlite metadata + local-file
    artifacts is the intended default here — see docs/methodology.md once
    written), so the experiment is created explicitly with that location the
    first time it's seen.
    """
    import mlflow

    existing = mlflow.get_experiment_by_name(experiment_name)
    if existing is None:
        artifact_location = f"file:{artifacts_root / 'mlflow' / experiment_name}"
        mlflow.create_experiment(experiment_name, artifact_location=artifact_location)
    mlflow.set_experiment(experiment_name)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _debug_truncate(manifest: pd.DataFrame, n_per_split: int) -> pd.DataFrame:
    parts = []
    for split in (SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST):
        subset = manifest[manifest["split"] == split]
        parts.append(subset.head(n_per_split))
    return pd.concat(parts, ignore_index=True)


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> float:
    """One pass over `loader`. Trains (backward+step) if `optimizer` is given,
    otherwise runs in eval mode. Returns mean loss for the epoch.
    """
    criterion = nn.BCEWithLogitsLoss()
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    n_batches = 0
    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            logits = model(images).squeeze(-1)
            loss = criterion(logits, labels)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += float(loss.item())
            n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def _collect_logits(model: nn.Module, loader: DataLoader, *, device: torch.device):
    model.eval()
    all_logits, all_labels = [], []
    for batch in loader:
        images = batch["image"].to(device)
        logits = model(images).squeeze(-1).cpu().numpy()
        all_logits.append(logits)
        all_labels.append(batch["label"].numpy())
    return np.concatenate(all_logits), np.concatenate(all_labels)


def run_training(cfg) -> dict:
    """Execute one full training run per `cfg`. Returns a summary dict
    (final val metrics, checkpoint paths, mlflow run id) — used directly by
    tests/smoke tests without needing to shell out to the CLI.
    """
    set_seed(int(cfg.training.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    manifest = load_manifest(cfg.data.manifest_path)
    check_no_leakage(manifest)

    if bool(cfg.training.get("debug", False)):
        manifest = _debug_truncate(manifest, int(cfg.training.get("debug_n_samples", 8)))

    train_manifest = manifest[manifest["split"] == SPLIT_TRAIN]
    val_manifest = manifest[manifest["split"] == SPLIT_VAL]
    if len(train_manifest) == 0:
        raise ValueError("no train-split samples after filtering — check manifest/debug_n_samples")
    if len(val_manifest) == 0:
        raise ValueError("no val-split samples after filtering — check manifest/debug_n_samples")

    image_size = int(cfg.data.get("image_size", 224))
    train_ds = ManifestImageDataset(train_manifest, image_size=image_size, train=True)
    val_ds = ManifestImageDataset(val_manifest, image_size=image_size, train=False)

    batch_size = int(cfg.training.batch_size)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = build_model(cfg.model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.training.lr))

    run_id = str(cfg.run_name)
    artifacts_root = Path(cfg.get("artifacts_root", REPO_ROOT / "artifacts"))
    ckpt_dir = checkpoint_dir(artifacts_root, run_id)

    start_epoch = 0
    global_step = 0
    best_metric = -float("inf")
    if cfg.training.get("resume") == "auto":
        latest = find_latest_checkpoint(artifacts_root, run_id)
        if latest is not None:
            state = load_checkpoint(latest, map_location=str(device))
            model.load_state_dict(state.model_state)
            optimizer.load_state_dict(state.optimizer_state)
            start_epoch = state.epoch + 1
            global_step = state.global_step
            best_metric = state.best_metric

    import mlflow

    configured_uri = cfg.get("mlflow_tracking_uri", None)
    tracking_uri = (
        str(configured_uri)
        if configured_uri
        else f"sqlite:///{artifacts_root / 'mlflow.db'}"
    )
    mlflow.set_tracking_uri(tracking_uri)
    _set_experiment(str(cfg.experiment_name), artifacts_root)

    val_metrics = None
    with mlflow.start_run(run_name=run_id) as run:
        mlflow.log_params(flatten_for_logging(cfg))

        epochs = int(cfg.training.epochs)
        for epoch in range(start_epoch, epochs):
            train_loss = _run_epoch(model, train_loader, device=device, optimizer=optimizer)
            val_loss = _run_epoch(model, val_loader, device=device, optimizer=None)
            global_step += len(train_loader)

            val_logits, val_labels = _collect_logits(model, val_loader, device=device)
            val_probs = expit(val_logits)
            val_metrics = compute_metrics(val_labels, val_probs)

            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    **{f"val_{k}": v for k, v in val_metrics.to_dict().items() if k != "n_samples"},
                },
                step=epoch,
            )

            current_metric = val_metrics.f1
            save_checkpoint(
                ckpt_dir / "latest.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                global_step=global_step,
                best_metric=max(best_metric, current_metric),
            )
            if current_metric > best_metric:
                best_metric = current_metric
                save_checkpoint(
                    ckpt_dir / "best.pt",
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    global_step=global_step,
                    best_metric=best_metric,
                )

        # Calibration + abstain-margin sweep, fit on the val set only.
        val_logits, val_labels = _collect_logits(model, val_loader, device=device)
        calibration = fit_temperature(val_logits, val_labels)
        calibration.save(ckpt_dir / "temperature.json")

        calibrated_val_probs = calibration.apply(val_logits)
        abstain_sweep = sweep_abstain_margin(calibrated_val_probs, val_labels)
        for row in abstain_sweep:
            mlflow.log_metric(f"abstain_sweep_acc_margin_{row['margin']}", row["accuracy_on_non_abstained"])
        mlflow.log_metric("calibration_temperature", calibration.temperature)

        mlflow.log_artifact(str(ckpt_dir / "latest.pt"))
        if (ckpt_dir / "best.pt").exists():
            mlflow.log_artifact(str(ckpt_dir / "best.pt"))
        mlflow.log_artifact(str(ckpt_dir / "temperature.json"))

        resolved_config_path = ckpt_dir / "resolved_config.yaml"
        from omegaconf import OmegaConf

        resolved_config_path.write_text(OmegaConf.to_yaml(cfg))
        mlflow.log_artifact(str(resolved_config_path))

        mlflow_run_id = run.info.run_id

    return {
        "run_id": run_id,
        "mlflow_run_id": mlflow_run_id,
        "val_metrics": val_metrics.to_dict() if val_metrics else None,
        "checkpoint_path": str(ckpt_dir / "latest.pt"),
        "best_checkpoint_path": str(ckpt_dir / "best.pt") if (ckpt_dir / "best.pt").exists() else None,
        "calibration_path": str(ckpt_dir / "temperature.json"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a YAML training config")
    args, overrides = parser.parse_known_args(argv)

    cfg = load_config(args.config, cli_overrides=overrides)
    summary = run_training(cfg)

    print(f"Training complete. run_id={summary['run_id']} mlflow_run_id={summary['mlflow_run_id']}")
    print(f"val_metrics={summary['val_metrics']}")
    print(f"checkpoint={summary['checkpoint_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
