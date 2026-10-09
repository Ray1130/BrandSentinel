"""End-to-end detection on the repository mock across crisis and rating-only scenarios."""

from datetime import timedelta

import polars as pl

from brandsentinel.core.schemas import validate
from brandsentinel.core.types import Table
from brandsentinel.detection.stage import detect
from brandsentinel.testing.mock_data import make_all

LAST = 14
SEEDS = range(8)
CRISIS_SKU = "B0MOCK0000"
RATING_ONLY_SKU = "B0MOCK0001"
NORMAL_SKUS = ("B0MOCK0002", "B0MOCK0003", "B0MOCK0004")
CHECKED_INDICATORS = ("I1", "I2", "I4", "I5", "I6")


def rate(matrix, sku, indicator, *, start=None, end=None):
    rows = matrix.filter(
        (pl.col("sku") == sku)
        & (pl.col("indicator_id") == indicator)
        & (pl.col("window_end") >= start if start else pl.lit(True))
        & (pl.col("window_end") < end if end else pl.lit(True))
    )
    return rows["triggered"].mean()


def test_detect_end_to_end_on_mock(cfg):
    rates = {}
    normal_hits = {indicator: [] for indicator in CHECKED_INDICATORS}
    normal_before = {indicator: [] for indicator in CHECKED_INDICATORS}

    for seed in SEEDS:
        feature_series = make_all(
            seed=seed,
            n_skus=5,
            days=120,
            cfg=cfg,
            rating_only_sku_index=1,
            bombing_sku_index=None,
        )[Table.FEATURE_SERIES]
        matrix = detect(feature_series, cfg)
        validate(Table.INDICATOR_MATRIX, matrix)

        assert set(matrix["indicator_id"].unique()) == {
            "I1",
            "I2",
            "I3",
            "I4",
            "I5",
            "I6",
        }
        assert matrix.group_by(["sku", "window_end", "indicator_id"]).len()["len"].max() == 1
        assert matrix["strength"].min() >= 0
        assert matrix["strength"].max() <= 1

        dates = feature_series.select("sku", "window_end").unique().sort(["sku", "window_end"])
        gaps = dates.with_columns(
            pl.col("window_end").diff().over("sku").dt.total_days().alias("gap")
        )["gap"].drop_nulls()
        assert gaps.min() == gaps.max() == 1
        assert feature_series["n_window"].null_count() == 0
        assert feature_series["window_end"].dtype == pl.Date
        assert dates.group_by("sku").len()["len"].min() >= 60

        baseline_end = dates["window_end"].min() + timedelta(days=59)
        crisis_start = dates["window_end"].max() - timedelta(days=LAST - 1)
        assert baseline_end < crisis_start
        last_start = dates["window_end"].max() - timedelta(days=LAST - 1)

        for indicator in CHECKED_INDICATORS:
            rates.setdefault(indicator, []).append(
                (
                    rate(matrix, CRISIS_SKU, indicator, start=last_start),
                    rate(matrix, RATING_ONLY_SKU, indicator, start=last_start),
                )
            )
            for sku in NORMAL_SKUS:
                normal_hits[indicator].append(rate(matrix, sku, indicator, start=last_start) > 0)
                normal_before[indicator].append(rate(matrix, sku, indicator, end=last_start))

    print("\nindicator | crisis S0 | rating-only S1 | normal SKU-seeds firing")
    for indicator in CHECKED_INDICATORS:
        crisis_and_rating = rates[indicator]
        print(
            indicator,
            round(sum(row[0] for row in crisis_and_rating) / len(crisis_and_rating), 3),
            round(sum(row[1] for row in crisis_and_rating) / len(crisis_and_rating), 3),
            round(sum(normal_hits[indicator]) / len(normal_hits[indicator]), 3),
        )

    for indicator in ("I1", "I2"):
        assert sum(row[0] for row in rates[indicator]) / len(rates[indicator]) > 0.3
        assert sum(row[1] for row in rates[indicator]) / len(rates[indicator]) <= 0.05
    for indicator in ("I4", "I5", "I6"):
        assert sum(row[0] for row in rates[indicator]) / len(rates[indicator]) > 0.3
        assert sum(row[1] for row in rates[indicator]) / len(rates[indicator]) > 0.3
    for indicator in CHECKED_INDICATORS:
        assert sum(normal_hits[indicator]) / len(normal_hits[indicator]) <= 0.15
        assert sum(normal_before[indicator]) / len(normal_before[indicator]) <= 0.05
