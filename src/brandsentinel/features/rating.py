"""Rating-related rolling features I4-I6."""

from __future__ import annotations

import numpy as np
import polars as pl

from brandsentinel.core.schemas import conform, empty_table, validate
from brandsentinel.core.types import Table
from brandsentinel.features.windows import rolling_sum, validate_window_days
from brandsentinel.preprocessing.validate import validate_daily_agg


def make_rating_features(daily_agg: pl.DataFrame, *, window_days: int = 7) -> pl.DataFrame:
	"""Build I4-I6 feature_series rows from a validated daily_agg table."""
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
		one_star = group["n1"].to_numpy().astype(np.float64)
		five_star = group["n5"].to_numpy().astype(np.float64)
		rating_sum = group["sum_rating"].to_numpy().astype(np.float64)
		squared_rating_sum = group["sumsq_rating"].to_numpy().astype(np.float64)

		window_count = rolling_sum(counts, window_days)
		window_negative = rolling_sum(negatives, window_days)
		window_one_star = rolling_sum(one_star, window_days)
		window_five_star = rolling_sum(five_star, window_days)
		window_rating_sum = rolling_sum(rating_sum, window_days)
		window_squared_rating_sum = rolling_sum(squared_rating_sum, window_days)

		for index in range(window_days - 1, len(dates)):
			denominator = window_count[index]
			variance = (
				float(
					window_squared_rating_sum[index] / denominator
					- (window_rating_sum[index] / denominator) ** 2
				)
				if denominator >= 2
				else None
			)
			values: tuple[tuple[str, str, float | None], ...] = (
				(
					"I4",
					"low_star_ratio_shrunk",
					float((window_negative[index] + 2) / (denominator + 20)),
				),
				("I5", "rating_rolling_variance", variance),
				(
					"I6",
					"extreme_share",
					float((window_one_star[index] + window_five_star[index]) / denominator)
					if denominator > 0
					else None,
				),
				(
					"I6",
					"share_1star",
					float(window_one_star[index] / denominator) if denominator > 0 else None,
				),
				(
					"I6",
					"share_5star",
					float(window_five_star[index] / denominator) if denominator > 0 else None,
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
						"n_window": int(denominator),
					}
				)

	result = conform(
		Table.FEATURE_SERIES,
		pl.DataFrame(rows, schema_overrides={"raw_value": pl.Float64})
		if rows
		else empty_table(Table.FEATURE_SERIES),
	)
	return validate(Table.FEATURE_SERIES, result)
