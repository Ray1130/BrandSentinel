"""Unit and integration tests for rolling content features (I7-I9) - Stage 2."""

from __future__ import annotations

import math
from datetime import date

import polars as pl
import pytest

from brandsentinel.core.schemas import validate
from brandsentinel.core.types import ASPECTS, Table
from brandsentinel.features.content import make_content_features
from brandsentinel.features.nlp.aspect import compute_aspect_window_features
from brandsentinel.features.nlp.risk_lexicon import compute_risk_window_features


def test_risk_window_features_deduplication():
    dates = [date(2023, 1, 1 + i) for i in range(7)]

    # Case 1: 1 spammer posting 4 risk reviews in the window
    reviews_spammer = pl.DataFrame(
        {
            "date": [date(2023, 1, 2), date(2023, 1, 3), date(2023, 1, 4), date(2023, 1, 5)],
            "user_id": ["same_user", "same_user", "same_user", "same_user"],
            "risk_hit": [True, True, True, True],
        }
    )

    rows_spammer = compute_risk_window_features(
        reviews_spammer,
        dates,
        sku="SKU_TEST",
        category="Baby_Products",
        window_days=7,
        window_counts=[4],
    )

    hits_row = next(r for r in rows_spammer if r["feature"] == "risk_hits")
    users_row = next(r for r in rows_spammer if r["feature"] == "risk_distinct_users")

    assert hits_row["raw_value"] == 4.0
    assert users_row["raw_value"] == 1.0  # Deduplicated down to 1 user

    # Case 2: 4 distinct users posting 1 risk review each
    reviews_distinct = pl.DataFrame(
        {
            "date": [date(2023, 1, 2), date(2023, 1, 3), date(2023, 1, 4), date(2023, 1, 5)],
            "user_id": ["u1", "u2", "u3", "u4"],
            "risk_hit": [True, True, True, True],
        }
    )

    rows_distinct = compute_risk_window_features(
        reviews_distinct,
        dates,
        sku="SKU_TEST",
        category="Baby_Products",
        window_days=7,
        window_counts=[4],
    )

    hits_row_2 = next(r for r in rows_distinct if r["feature"] == "risk_hits")
    users_row_2 = next(r for r in rows_distinct if r["feature"] == "risk_distinct_users")

    assert hits_row_2["raw_value"] == 4.0
    assert users_row_2["raw_value"] == 4.0  # 4 distinct users!


def test_aspect_window_features_ratio_and_null():
    dates = [date(2023, 1, 1 + i) for i in range(7)]

    # 3 reviews for quality: 2 negative, 1 positive.
    # 0 reviews for delivery, safety, refund.
    reviews_df = pl.DataFrame(
        {
            "date": [date(2023, 1, 2), date(2023, 1, 3), date(2023, 1, 4)],
            "aspect": ["quality", "quality", "quality"],
            "aspect_sentiment": [-0.8, -0.5, 0.9],
            "sentiment_score": [-0.8, -0.5, 0.9],
        }
    )

    rows = compute_aspect_window_features(
        reviews_df,
        dates,
        sku="SKU_TEST",
        category="Baby_Products",
        window_days=7,
        window_counts=[3],
    )

    assert len(rows) == 4  # 4 aspects in ASPECTS
    row_quality = next(r for r in rows if r["feature"] == "aspect_neg_quality")
    row_delivery = next(r for r in rows if r["feature"] == "aspect_neg_delivery")

    assert pytest.approx(row_quality["raw_value"], 0.01) == 2.0 / 3.0
    # Delivery has 0 reviews -> MUST be None (Polars null), NOT NaN!
    assert row_delivery["raw_value"] is None


def test_make_content_features_with_mock_tables(mock_tables, cfg):
    clean = mock_tables[Table.CLEAN_REVIEWS]
    daily = mock_tables[Table.DAILY_AGG]
    nlp = mock_tables[Table.NLP_FEATURES]

    content_feats = make_content_features(
        clean,
        nlp,
        daily,
        window_days=cfg.default.time.window_days,
    )

    # 1. Output must pass strict Table.FEATURE_SERIES schema validation
    validated = validate(Table.FEATURE_SERIES, content_feats)
    assert validated.height > 0

    # 2. Check generated indicators
    unique_inds = set(content_feats["indicator_id"].unique().to_list())
    assert {"I7", "I8", "I9"}.issubset(unique_inds)

    # 3. Check generated features
    unique_feats = set(content_feats["feature"].unique().to_list())
    expected_feats = {
        "term_burst_score",
        "risk_hits",
        "risk_distinct_users",
        "aspect_neg_quality",
        "aspect_neg_delivery",
        "aspect_neg_safety",
        "aspect_neg_refund",
    }
    assert expected_feats.issubset(unique_feats)

    # 4. Check that no raw_value is NaN or inf
    raw_vals = content_feats["raw_value"].drop_nulls().to_list()
    for val in raw_vals:
        assert not math.isnan(val)
        assert not math.isinf(val)


def test_make_content_features_empty_handling():
    empty_clean = pl.DataFrame(schema={"review_id": pl.String, "is_spam": pl.Boolean})
    empty_nlp = pl.DataFrame(schema={"review_id": pl.String})

    result = make_content_features(empty_clean, empty_nlp)
    assert result.is_empty()
    validate(Table.FEATURE_SERIES, result)
