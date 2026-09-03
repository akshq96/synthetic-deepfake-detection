"""Robustness curve plot: accuracy (and ROC-AUC) vs perturbation severity,
one line per perturbation type, with the baseline (unperturbed) accuracy
drawn as a reference line — the plot Phase 6 exists to produce.
"""

from __future__ import annotations

import csv
from pathlib import Path

from ml.plotting.common import ensure_parent, use_non_interactive_backend


def plot_robustness_curve(
    results: dict, output_path: str | Path, *, metric: str = "accuracy"
) -> Path:
    """`results` is the dict returned by
    ml.evaluation.experiments.robustness.run_robustness_experiment. Writes
    `output_path` (.png) and a sibling .csv with the same data. Returns the
    .png path.
    """
    use_non_interactive_backend()
    import matplotlib.pyplot as plt

    output_path = ensure_parent(output_path)
    baseline_value = results["baseline_metrics"][metric]

    fig, ax = plt.subplots(figsize=(7, 5))
    csv_rows = [("perturbation", "severity", metric)]

    for pert_name, curve in results["perturbations"].items():
        severities = [point["severity"] for point in curve]
        values = [point["metrics"][metric] for point in curve]
        ax.plot(severities, values, marker="o", label=pert_name)
        csv_rows += [(pert_name, s, v) for s, v in zip(severities, values)]

    ax.axhline(baseline_value, color="gray", linestyle="--", label="baseline (unperturbed)")
    ax.set_xlabel("perturbation severity")
    ax.set_ylabel(metric)
    ax.set_title(f"Robustness: {metric} vs. perturbation severity")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    csv_path = output_path.with_suffix(".csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)

    return output_path
