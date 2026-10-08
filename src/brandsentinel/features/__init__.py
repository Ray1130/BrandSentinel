"""Feature generation from validated preprocessing tables."""

from __future__ import annotations

import polars as pl

from brandsentinel.core.schemas import empty_table, validate
from brandsentinel.core.types import Table
from brandsentinel.features.rating import make_rating_features
from brandsentinel.features.volume import make_volume_features
from brandsentinel.features.windows import validate_window_days
from brandsentinel.preprocessing.validate import validate_daily_agg


def make_feature_series(daily_agg: pl.DataFrame, *, window_days: int = 7) -> pl.DataFrame:
	"""Build and validate volume/rating feature_series rows for indicators I1-I6."""
	validate_window_days(window_days)
	validate_daily_agg(daily_agg)
	if daily_agg.is_empty():
		return validate(Table.FEATURE_SERIES, empty_table(Table.FEATURE_SERIES))

	result = pl.concat(
		[
			make_volume_features(daily_agg, window_days=window_days),
			make_rating_features(daily_agg, window_days=window_days),
		]
	)
	return validate(Table.FEATURE_SERIES, result)
