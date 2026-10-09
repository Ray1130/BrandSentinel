"""Comprehensive integration tests for full feature_series generation (I1-I9) - Stage 4."""

from __future__ import annotations

import math
from datetime import date

import polars as pl
import pytest

from brandsentinel.core.schemas import validate
from brandsentinel.core.types import ASPECTS, Table
from brandsentinel.features import make_feature_series
from brandsentinel.testing.mock_data import make_feature_series as make_mock_feature_series


def test_make_feature_series_full_pipeline(mock_tables, cfg):
    window_days = cfg.default.time.window_days
    clean = mock_tables[Table.CLEAN_REVIEWS]
    daily = mock_tables[Table.DAILY_AGG]
    nlp = mock_tables[Table.NLP_FEATURES]

    # Chạy full pipeline tạo I1-I9
    feats = make_feature_series(clean, daily, nlp, window_days=window_days)

    # 1. Hợp đồng dữ liệu hợp lệ
    validated = validate(Table.FEATURE_SERIES, feats)
    assert validated.height > 0

    # 2. Đầy đủ cả 9 chỉ báo cốt lõi I1-I9
    indicators = set(feats["indicator_id"].unique().to_list())
    expected_indicators = {f"I{i}" for i in range(1, 10)}
    assert expected_indicators.issubset(indicators)

    # 3. Đầy đủ các đặc trưng nội dung I7, I8, I9
    features = set(feats["feature"].unique().to_list())
    expected_content_features = {
        "term_burst_score",
        "risk_hits",
        "risk_distinct_users",
        "aspect_neg_quality",
        "aspect_neg_delivery",
        "aspect_neg_safety",
        "aspect_neg_refund",
    }
    assert expected_content_features.issubset(features)

    # 4. Kiểm tra triệt tiêu NaN và Inf
    raw_values = feats["raw_value"].drop_nulls().to_list()
    for val in raw_values:
        assert not math.isnan(val), "Phát hiện giá trị NaN trong raw_value!"
        assert not math.isinf(val), "Phát hiện giá trị Inf trong raw_value!"

    # 5. Kiểm tra n_window không âm
    assert (feats["n_window"] >= 0).all()


def test_make_feature_series_calling_flexibility(mock_tables, cfg):
    window_days = cfg.default.time.window_days
    clean = mock_tables[Table.CLEAN_REVIEWS]
    daily = mock_tables[Table.DAILY_AGG]
    nlp = mock_tables[Table.NLP_FEATURES]

    # Kiểu 1: Chỉ truyền daily -> I1-I6
    feats_vol_rat = make_feature_series(daily, window_days=window_days)
    inds_1 = set(feats_vol_rat["indicator_id"].unique().to_list())
    assert inds_1 == {f"I{i}" for i in range(1, 7)}

    # Kiểu 2: Chỉ truyền clean + nlp -> I7-I9
    feats_content = make_feature_series(clean_reviews=clean, nlp_features=nlp, window_days=window_days)
    inds_2 = set(feats_content["indicator_id"].unique().to_list())
    assert inds_2 == {"I7", "I8", "I9"}

    # Kiểu 3: Truyền keyword arguments đầy đủ -> I1-I9
    feats_full_kw = make_feature_series(
        daily_agg=daily, clean_reviews=clean, nlp_features=nlp, window_days=window_days
    )
    inds_3 = set(feats_full_kw["indicator_id"].unique().to_list())
    assert {f"I{i}" for i in range(1, 10)}.issubset(inds_3)

    # Kiểu 4: Không truyền gì -> Empty table hợp lệ
    empty_feats = make_feature_series()
    assert empty_feats.is_empty()
    validate(Table.FEATURE_SERIES, empty_feats)


def test_feature_series_matches_mock_reference(mock_tables, cfg):
    window_days = cfg.default.time.window_days
    clean = mock_tables[Table.CLEAN_REVIEWS]
    daily = mock_tables[Table.DAILY_AGG]
    nlp = mock_tables[Table.NLP_FEATURES]

    actual = make_feature_series(clean, daily, nlp, window_days=window_days)
    expected = make_mock_feature_series(clean, daily, nlp, window_days=window_days)

    # So sánh I8 (risk_hits và risk_distinct_users)
    sort_cols = ["sku", "window_end", "indicator_id", "feature"]
    actual_i8 = actual.filter(pl.col("indicator_id") == "I8").sort(sort_cols)
    expected_i8 = expected.filter(pl.col("indicator_id") == "I8").sort(sort_cols)
    assert actual_i8.equals(expected_i8)

    # So sánh I9 (aspect_neg_*)
    actual_i9 = actual.filter(pl.col("indicator_id") == "I9").sort(sort_cols)
    expected_i9 = expected.filter(pl.col("indicator_id") == "I9").sort(sort_cols)
    assert actual_i9.equals(expected_i9)
