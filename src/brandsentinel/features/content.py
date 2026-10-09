"""Content-related rolling window features I7-I9."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from brandsentinel.core.schemas import conform, empty_table, validate
from brandsentinel.core.types import Table
from brandsentinel.features.nlp.aspect import compute_aspect_window_features
from brandsentinel.features.nlp.risk_lexicon import compute_risk_window_features
from brandsentinel.features.nlp.topic import compute_topic_window_features
from brandsentinel.features.windows import rolling_sum, validate_window_days


def make_content_features(
    clean_reviews: pl.DataFrame,
    nlp_features: pl.DataFrame,
    daily_agg: pl.DataFrame | None = None,
    *,
    window_days: int = 7,
) -> pl.DataFrame:
    """Xây dựng và kiểm thực các dòng feature_series cho nhóm nội dung I7-I9.

    Args:
        clean_reviews: Bảng clean_reviews đã kiểm thực.
        nlp_features: Bảng nlp_features đã kiểm thực.
        daily_agg: Bảng daily_agg tùy chọn để đồng bộ các mốc ngày và n_window.
        window_days: Kích thước cửa sổ trượt (ngày, mặc định 7).

    Returns:
        DataFrame feature_series hợp lệ chứa các dòng I7, I8, I9.
    """
    validate_window_days(window_days)
    if clean_reviews.is_empty() or nlp_features.is_empty():
        return validate(Table.FEATURE_SERIES, empty_table(Table.FEATURE_SERIES))

    # Lọc bỏ review spam
    non_spam = clean_reviews.filter(~pl.col("is_spam"))
    if non_spam.is_empty():
        return validate(Table.FEATURE_SERIES, empty_table(Table.FEATURE_SERIES))

    # Ghép bảng clean_reviews và nlp_features theo review_id
    nlp_cols = [
        c
        for c in [
            "review_id",
            "risk_hit",
            "aspect",
            "aspect_sentiment",
            "sentiment_score",
        ]
        if c in nlp_features.columns
    ]
    joined = non_spam.join(nlp_features.select(nlp_cols), on="review_id", how="inner")
    if joined.is_empty():
        return validate(Table.FEATURE_SERIES, empty_table(Table.FEATURE_SERIES))

    rows: list[dict[str, Any]] = []

    # Map bảng daily_agg theo từng SKU nếu có
    daily_by_sku: dict[str, pl.DataFrame] = {}
    if daily_agg is not None and not daily_agg.is_empty():
        for (sku,), g in daily_agg.sort("date").partition_by(["sku"], as_dict=True).items():
            daily_by_sku[sku] = g

    for (sku,), sku_reviews in joined.sort("date").partition_by(["sku"], as_dict=True).items():
        category = sku_reviews["category"][0]

        if sku in daily_by_sku:
            g_daily = daily_by_sku[sku]
            dates = g_daily["date"].to_list()
            counts = g_daily["n"].to_numpy().astype(np.float64)
            window_counts_arr = rolling_sum(counts, window_days)
            window_counts = [int(c) if not np.isnan(c) else 0 for c in window_counts_arr]
        else:
            # Fallback nếu không truyền daily_agg: tạo dải ngày liên tục theo lịch
            min_d, max_d = sku_reviews["date"].min(), sku_reviews["date"].max()
            date_df = pl.DataFrame({"date": pl.date_range(min_d, max_d, "1d", eager=True)})
            day_counts = (
                date_df.join(
                    sku_reviews.group_by("date").agg(pl.len().alias("n")),
                    on="date",
                    how="left",
                )
                .fill_null(0)
                .sort("date")
            )
            dates = day_counts["date"].to_list()
            counts = day_counts["n"].to_numpy().astype(np.float64)
            window_counts_arr = rolling_sum(counts, window_days)
            window_counts = [int(c) if not np.isnan(c) else 0 for c in window_counts_arr]

        if len(dates) < window_days:
            continue

        # I7: term_burst_score
        rows.extend(
            compute_topic_window_features(
                sku_reviews,
                dates,
                sku=sku,
                category=category,
                window_days=window_days,
                window_counts=window_counts,
            )
        )

        # I8: risk_hits & risk_distinct_users
        rows.extend(
            compute_risk_window_features(
                sku_reviews,
                dates,
                sku=sku,
                category=category,
                window_days=window_days,
                window_counts=window_counts,
            )
        )

        # I9: aspect_neg_*
        rows.extend(
            compute_aspect_window_features(
                sku_reviews,
                dates,
                sku=sku,
                category=category,
                window_days=window_days,
                window_counts=window_counts,
            )
        )

    result = conform(
        Table.FEATURE_SERIES,
        pl.DataFrame(rows, schema_overrides={"raw_value": pl.Float64})
        if rows
        else empty_table(Table.FEATURE_SERIES),
    )
    return validate(Table.FEATURE_SERIES, result)
