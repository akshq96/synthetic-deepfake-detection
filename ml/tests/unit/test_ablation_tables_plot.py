from __future__ import annotations

from pathlib import Path

from ml.plotting.ablation_tables import plot_ablation_curve, save_ablation_table


def _single_key_results() -> list[dict]:
    return [
        {"point": {"data.synthetic_ratio": 0.0}, "eval_metrics": {"accuracy": 0.7}},
        {"point": {"data.synthetic_ratio": 0.3}, "eval_metrics": {"accuracy": 0.8}},
        {"point": {"data.synthetic_ratio": 0.5}, "eval_metrics": {"accuracy": 0.75}},
    ]


def _multi_key_results() -> list[dict]:
    return [
        {"point": {"a": 1, "b": 3}, "eval_metrics": {"accuracy": 0.7}},
        {"point": {"a": 1, "b": 4}, "eval_metrics": {"accuracy": 0.8}},
    ]


def test_save_ablation_table_writes_csv(tmp_path: Path):
    path = save_ablation_table(_single_key_results(), tmp_path / "ablation.csv")
    assert path.exists()
    content = path.read_text()
    assert "data.synthetic_ratio" in content
    assert "accuracy" in content
    assert "0.8" in content


def test_save_ablation_table_multi_key(tmp_path: Path):
    path = save_ablation_table(_multi_key_results(), tmp_path / "ablation_multi.csv")
    content = path.read_text()
    assert "a" in content and "b" in content


def test_plot_ablation_curve_single_key(tmp_path: Path):
    path = plot_ablation_curve(_single_key_results(), tmp_path / "ablation.png")
    assert path is not None
    assert path.exists()
    assert path.stat().st_size > 0


def test_plot_ablation_curve_multi_key_returns_none(tmp_path: Path):
    path = plot_ablation_curve(_multi_key_results(), tmp_path / "ablation.png")
    assert path is None
    assert not (tmp_path / "ablation.png").exists()
