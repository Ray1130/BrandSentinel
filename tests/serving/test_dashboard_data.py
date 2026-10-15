from __future__ import annotations

from datetime import date, timedelta
from importlib.util import find_spec
from pathlib import Path

import polars as pl
import pytest

from brandsentinel.core.config import load_config
from brandsentinel.core.io import write_table
from brandsentinel.core.types import Table
from brandsentinel.llm.evidence import build_evidence_pack
from brandsentinel.serving import dashboard_data as dashboard


def test_priority_table_uses_level_score_and_i8_as_tie_breaker(cfg):
    alerts = pl.DataFrame(
        {
            "sku": ["A", "B", "C", "A", "B"],
            "category": ["test"] * 5,
            "window_end": [
                date(2024, 1, 1),
                date(2024, 1, 1),
                date(2024, 1, 1),
                date(2024, 1, 2),
                date(2024, 1, 2),
            ],
            "score": [0.2, 0.2, 0.3, 0.5, 0.5],
            "level": ["MEDIUM", "MEDIUM", "MEDIUM", "HIGH", "HIGH"],
            "groups_active": [["volume"], ["rating"], ["content"], ["volume"], ["content"]],
            "persist": [2, 2, 2, 1, 1],
            "verify_flag": [False, True, False, False, False],
            "triggered_ids": [["I1"], ["I4"], ["I7"], ["I1"], ["I8"]],
        }
    )

    ranked = dashboard.priority_table(alerts, date(2024, 1, 2))

    assert ranked["sku"].to_list() == ["B", "A", "C"]
    assert ranked.filter(pl.col("sku") == "A")["level_since"][0] == date(2024, 1, 2)
    assert dashboard.snapshot(alerts, date(2024, 1, 1))["sku"].to_list() == ["C", "A", "B"]


def test_indicator_contributions_match_score_and_exclude_future(cfg, mock_tables):
    alerts = mock_tables[Table.ALERTS]
    assert not alerts.is_empty()
    chosen = alerts.sort("window_end").row(-1, named=True)
    as_of = chosen["window_end"]
    matrix = mock_tables[Table.INDICATOR_MATRIX].filter(
        (pl.col("sku") == chosen["sku"]) & (pl.col("window_end") == as_of)
    )
    breakdown = dashboard.indicator_breakdown(matrix, cfg, chosen["sku"], as_of)

    assert sum(row["contribution"] for row in breakdown) == pytest.approx(chosen["score"])
    assert sum(row["weight"] for row in breakdown) == pytest.approx(1.0)
    assert all(row["group"] != "flag" or row["contribution"] == 0 for row in breakdown)
    assert dashboard.snapshot(alerts, as_of + timedelta(days=1))["window_end"].max() == as_of


def test_evidence_reviews_match_llm_selection_and_window(cfg, mock_tables, monkeypatch):
    clean = mock_tables[Table.CLEAN_REVIEWS]
    nlp = mock_tables[Table.NLP_FEATURES]
    sku = clean["sku"][0]
    end = clean.filter(pl.col("sku") == sku)["date"].max()
    record = {
        "sku": sku,
        "category": clean["category"][0],
        "window_end": end,
        "score": 0.5,
        "level": "HIGH",
        "groups_active": ["volume", "rating", "content"],
        "persist": 2,
        "verify_flag": False,
        "triggered_ids": ["I8"],
    }

    def fake_read(table, _config, _category, *, start=None, end=None, skus=None, review_ids=None):
        if table == Table.CLEAN_REVIEWS:
            return clean.filter(
                (pl.col("sku") == sku) & pl.col("date").is_between(start, end, closed="both")
            )
        return nlp.filter(pl.col("review_id").is_in(review_ids or []))

    monkeypatch.setattr(dashboard, "read_category_table", fake_read)
    actual = dashboard.evidence_reviews(cfg, record["category"], record)
    window_start = end - timedelta(days=cfg.default.time.window_days - 1)
    clean_window = clean.filter(
        (pl.col("sku") == sku) & pl.col("date").is_between(window_start, end, closed="both")
    )
    expected = build_evidence_pack(
        record,
        clean_window,
        nlp_features_df=nlp,
        window_days=cfg.default.time.window_days,
        max_reviews=cfg.llm.max_top_reviews,
        mask_pii_enabled=getattr(cfg.llm, "mask_pii", True),
    )

    assert {review["review_id"] for review in actual} == expected.review_ids_set
    assert all(window_start.isoformat() <= review["date"] <= end.isoformat() for review in actual)


def test_missing_recommendation_is_a_visible_status(cfg, tmp_path):
    config = cfg.model_copy(update={"root": tmp_path})

    result = dashboard.load_recommendation(config, "No_Category", "SKU", date(2024, 1, 1))

    assert result["status"] == "missing"
    assert result["recommendation"] is None


@pytest.mark.skipif(
    find_spec("streamlit") is None, reason="streamlit serving extra is not installed"
)
def test_dashboard_app_smoke_on_empty_data():
    from streamlit.testing.v1 import AppTest

    dashboard_file = (
        Path(__file__).parents[2] / "src" / "brandsentinel" / "serving" / "dashboard.py"
    )
    result = AppTest.from_file(str(dashboard_file)).run()

    assert not result.exception


@pytest.mark.skipif(
    find_spec("streamlit") is None, reason="streamlit serving extra is not installed"
)
def test_dashboard_app_smoke_on_mock_data(mock_tables, tmp_path, monkeypatch):
    import shutil

    from streamlit.testing.v1 import AppTest

    repo = Path(__file__).parents[2]
    config_dir = tmp_path / "configs"
    shutil.copytree(repo / "configs", config_dir)
    shutil.copy2(repo / "configs" / "llm.yaml", tmp_path / "llm.yaml")
    mock_cfg = load_config(config_dir)
    for table, frame in mock_tables.items():
        write_table(frame, table, mock_cfg)

    monkeypatch.setenv("BS_CONFIG_DIR", str(config_dir))
    from brandsentinel.core.config import get_config

    get_config.cache_clear()
    try:
        dashboard_file = repo / "src" / "brandsentinel" / "serving" / "dashboard.py"
        result = AppTest.from_file(str(dashboard_file)).run()
        assert not result.exception
    finally:
        get_config.cache_clear()
