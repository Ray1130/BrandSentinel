import polars as pl

from brandsentinel.core.types import Table
from brandsentinel.features import make_feature_series
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