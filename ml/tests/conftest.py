from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ml.data_pipeline.split import assign_splits
from ml.tests.fixtures.synthetic_fixture import generate_fixture_dataset


@pytest.fixture
def fixture_manifest_unsplit(tmp_path: Path) -> pd.DataFrame:
    return generate_fixture_dataset(tmp_path)


@pytest.fixture
def fixture_manifest_split(fixture_manifest_unsplit: pd.DataFrame) -> pd.DataFrame:
    return assign_splits(fixture_manifest_unsplit, seed=42)
