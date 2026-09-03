"""Phase 1 smoke test: exercises the whole dataset pipeline slice built so far
(manifest build -> save/load -> leakage-safe split -> leakage check) against
the tiny synthetic fixture dataset, end to end, asserting each stage produces
the expected output. Later phases extend this file (or add a sibling) to
continue the chain through training/eval/XAI, per the project plan.
"""

from __future__ import annotations

from pathlib import Path

from ml.data_pipeline.leakage_check import check_no_leakage
from ml.data_pipeline.manifest import load_manifest, save_manifest
from ml.data_pipeline.split import assign_splits
from ml.data_pipeline.schema import validate_manifest
from ml.tests.fixtures.synthetic_fixture import generate_fixture_dataset, total_fixture_samples


def test_dataset_pipeline_end_to_end(tmp_path: Path):
    # 1. Generate fixture images + build manifest.
    manifest = generate_fixture_dataset(tmp_path / "raw")
    assert len(manifest) == total_fixture_samples()
    for p in manifest["file_path"]:
        assert Path(p).exists()

    # 2. Persist and reload the manifest (parquet round-trip).
    manifest_path = tmp_path / "manifests" / "fixture.parquet"
    save_manifest(manifest, manifest_path)
    reloaded = load_manifest(manifest_path)
    assert len(reloaded) == len(manifest)

    # 3. Assign leakage-safe splits.
    split_manifest = assign_splits(reloaded, seed=42)
    validate_manifest(split_manifest, require_split=True)
    assert set(split_manifest["split"].unique()) <= {"train", "val", "test"}
    assert split_manifest["split"].isna().sum() == 0

    # 4. Leakage check must pass on a correctly split manifest.
    check_no_leakage(split_manifest)

    # 5. Persist the final split manifest, as a real pipeline run would.
    split_manifest_path = tmp_path / "manifests" / "fixture_split.parquet"
    save_manifest(split_manifest, split_manifest_path)
    assert split_manifest_path.exists()
