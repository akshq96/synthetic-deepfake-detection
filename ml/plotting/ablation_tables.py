"""Ablation results -> a paper-ready table (CSV) and a simple line/bar plot
when the sweep varied exactly one numeric key (e.g. synthetic_ratio) — the
common case. For multi-key sweeps, only the CSV table is produced (a single
2-D plot can't represent an arbitrary-dimensional grid).
"""

from __future__ import annotations

import csv
from pathlib import Path

from ml.plotting.common import ensure_parent, use_non_interactive_backend


def save_ablation_table(results: list[dict], output_path: str | Path, *, metric: str = "accuracy") -> Path:
    output_path = ensure_parent(output_path)
    point_keys = sorted({k for r in results for k in r["point"].keys()})

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([*point_keys, metric])
        for r in results:
            row = [r["point"].get(k) for k in point_keys]
            row.append(r["eval_metrics"][metric])
            writer.writerow(row)

    return output_path


def plot_ablation_curve(
    results: list[dict], output_path: str | Path, *, metric: str = "accuracy"
) -> Path | None:
    """Returns the PNG path, or None if the sweep isn't single-key (nothing
    plotted in that case — call save_ablation_table regardless for the CSV).
    """
    point_keys = {k for r in results for k in r["point"].keys()}
    if len(point_keys) != 1:
        return None

    use_non_interactive_backend()
    import matplotlib.pyplot as plt

    (key,) = point_keys
    output_path = ensure_parent(output_path)

    sorted_results = sorted(results, key=lambda r: r["point"][key])
    x_values = [r["point"][key] for r in sorted_results]
    y_values = [r["eval_metrics"][metric] for r in sorted_results]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x_values, y_values, marker="o")
    ax.set_xlabel(key)
    ax.set_ylabel(metric)
    ax.set_title(f"Ablation: {metric} vs. {key}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path
