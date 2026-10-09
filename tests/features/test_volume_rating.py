import polars as pl

from brandsentinel.core.types import Table
from brandsentinel.features import make_feature_series
from brandsentinel.features.volume import make_volume_features
from brandsentinel.testing.mock_data import make_feature_series as make_mock_feature_series


def test_volume_rating_features_match_mock(mock_tables, cfg):
    window_days = cfg.default.time.window_days
    actual = make_feature_series(mock_tables[Table.DAILY_AGG], window_days=window_days)
    expected = make_mock_feature_series(
        mock_tables[Table.CLEAN_REVIEWS],
        mock_tables[Table.DAILY_AGG],
        mock_tables[Table.NLP_FEATURES],
        window_days=window_days,
    ).filter(pl.col("indicator_id").is_in([f"I{i}" for i in range(1, 7)]))

    sort_by = ["sku", "window_end", "indicator_id", "feature"]
    assert actual.sort(sort_by).equals(expected.sort(sort_by))


def test_growth_rate_compares_consecutive_full_windows(mock_tables, cfg):
    window_days = cfg.default.time.window_days
    daily = mock_tables[Table.DAILY_AGG].filter(pl.col("sku") == "B0MOCK0000").sort("date")
    counts = daily["n"].to_list()
    features = make_volume_features(daily, window_days=window_days).filter(
        pl.col("feature") == "growth_rate"
    )

    first_valid_index = 2 * window_days - 1
    previous = sum(
        counts[first_valid_index - 2 * window_days + 1 : first_valid_index - window_days + 1]
    )
    current = sum(counts[first_valid_index - window_days + 1 : first_valid_index + 1])
    expected = (current - previous) / (previous + 1.0)
    observed = features.filter(pl.col("window_end") == daily["date"][first_valid_index])[
        "raw_value"
    ][0]

    assert features["raw_value"].null_count() == window_days
    first_valid = features.filter(pl.col("raw_value").is_not_null())["window_end"][0]
    assert first_valid == daily["date"][first_valid_index]
    assert observed == expected
