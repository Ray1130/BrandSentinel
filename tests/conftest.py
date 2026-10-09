import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from brandsentinel.core.config import load_config
from brandsentinel.core.types import Table
from brandsentinel.testing.mock_data import make_all


@pytest.fixture(scope="session")
def cfg():
    return load_config(REPO / "configs")


@pytest.fixture(scope="session")
def mock_tables(cfg):
    return make_all(seed=42, cfg=cfg)


@pytest.fixture(scope="session")
def feature_series(mock_tables):
    return mock_tables[Table.FEATURE_SERIES]
