from __future__ import annotations

from pathlib import Path

from ml.plotting.robustness_curves import plot_robustness_curve


def _fake_results() -> dict:
    return {
        "baseline_metrics": {"accuracy": 0.9, "roc_auc": 0.95},
        "perturbations": {
            "gaussian_noise": [
                {"severity": 5.0, "metrics": {"accuracy": 0.88, "roc_auc": 0.93}},
                {"severity": 30.0, "metrics": {"accuracy": 0.6, "roc_auc": 0.65}},
            ],
            "compression": [
                {"severity": 80, "metrics": {"accuracy": 0.87, "roc_auc": 0.92}},
                {"severity": 20, "metrics": {"accuracy": 0.7, "roc_auc": 0.75}},
            ],
        },
    }


def test_plot_robustness_curve_writes_png_and_csv(tmp_path: Path):
    out_path = plot_robustness_curve(_fake_results(), tmp_path / "figures" / "robustness.png")
    assert out_path.exists()
    assert out_path.stat().st_size > 0

    csv_path = out_path.with_suffix(".csv")
    assert csv_path.exists()
    content = csv_path.read_text()
    assert "gaussian_noise" in content
    assert "compression" in content
    assert "0.88" in content


def test_plot_robustness_curve_supports_alternate_metric(tmp_path: Path):
    out_path = plot_robustness_curve(_fake_results(), tmp_path / "roc_auc.png", metric="roc_auc")
    assert out_path.exists()
    assert "0.93" in out_path.with_suffix(".csv").read_text()
