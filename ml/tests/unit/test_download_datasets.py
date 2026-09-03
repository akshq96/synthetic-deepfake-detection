from __future__ import annotations

from pathlib import Path

import ml.scripts.download_datasets as dd


def test_check_dataset_missing_directory(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.setattr(dd, "DATA_RAW", tmp_path)
    ok = dd.check_dataset("ffpp")
    assert ok is False
    out = capsys.readouterr().out
    assert "[MISSING]" in out
    assert "github.com/ondyari/FaceForensics" in out


def test_check_dataset_incomplete_directory(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.setattr(dd, "DATA_RAW", tmp_path)
    (tmp_path / "ffpp").mkdir()
    ok = dd.check_dataset("ffpp")
    assert ok is False
    assert "[INCOMPLETE]" in capsys.readouterr().out


def test_check_dataset_ok(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.setattr(dd, "DATA_RAW", tmp_path)
    root = tmp_path / "ffpp"
    (root / "original_sequences").mkdir(parents=True)
    (root / "manipulated_sequences").mkdir(parents=True)
    ok = dd.check_dataset("ffpp")
    assert ok is True
    assert "[OK]" in capsys.readouterr().out


def test_main_reports_summary_and_exit_code(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.setattr(dd, "DATA_RAW", tmp_path)
    code = dd.main(["--dataset", "celebdf"])
    assert code == 1
    assert "0/1 dataset(s) ready" in capsys.readouterr().out
