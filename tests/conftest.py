from pathlib import Path

import pytest

from brandsentinel.core.config import load_config
from brandsentinel.testing.mock_data import make_all

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def cfg():
    return load_config(REPO / "configs")


@pytest.fixture(scope="session")
def mock_tables(cfg):
    return make_all(seed=42, cfg=cfg)
