"""Volume-related daily features I1-I3."""

from __future__ import annotations

import math

import numpy as np
import polars as pl

from brandsentinel.core.schemas import conform, empty_table, validate
from brandsentinel.core.types import Table
from brandsentinel.features.windows import rolling_sum, validate_window_days
from brandsentinel.preprocessing.validate import validate_daily_agg


def make_volume_features(daily_agg: pl.DataFrame, *, window_days: int = 7) -> pl.DataFrame:
	"""Build I1-I3 feature_series rows from a validated daily_agg table."""
	validate_window_days(window_days)
	validate_daily_agg(daily_agg)
	if daily_agg.is_empty():
		return validate(Table.FEATURE_SERIES, empty_table(Table.FEATURE_SERIES))

	rows: list[dict[str, object]] = []
	for _, group in daily_agg.sort(["sku", "date"]).partition_by(
		"sku", as_dict=True
	).items():
		sku = group["sku"][0]
		category = group["category"][0]
		dates = group["date"].to_list()
		counts = group["n"].to_numpy().astype(np.float64)
		negatives = group["n_neg"].to_numpy().astype(np.float64)
		neutral = group["n3"].to_numpy().astype(np.float64)
		window_count = rolling_sum(counts, window_days)
		window_negative = rolling_sum(negatives, window_days)
		window_neutral = rolling_sum(neutral, window_days)

		for index in range(window_days - 1, len(dates)):
			prior_count = counts[index - 1] if index else None
			values: tuple[tuple[str, str, float | None], ...] = (
				("I1", "log1p_daily_count", math.log1p(counts[index])),
				(
					"I2",
					"growth_rate",
					(counts[index] - prior_count) / (prior_count + 1.0)
					if prior_count is not None
					else None,
				),
				(
					"I3",
					"neg_ratio_shrunk",
					float((window_negative[index] + 0.5 * window_neutral[index] + 2) / (window_count[index] + 20)),
				),
			)
			for indicator_id, feature, raw_value in values:
				rows.append(
					{
						"sku": sku,
						"category": category,
						"window_end": dates[index],
						"indicator_id": indicator_id,
						"feature": feature,
						"raw_value": raw_value,
						"n_window": int(window_count[index]),
					}
				)

	result = conform(
		Table.FEATURE_SERIES,
		pl.DataFrame(rows, schema_overrides={"raw_value": pl.Float64})
		if rows
		else empty_table(Table.FEATURE_SERIES),
	)
	return validate(Table.FEATURE_SERIES, result)
