"""Reliability features I10-I11."""

from __future__ import annotations

import polars as pl

from brandsentinel.core.schemas import conform, empty_table, validate
from brandsentinel.core.types import Table
from brandsentinel.preprocessing.validate import validate_daily_agg


def make_reliability_features(
	daily_agg: pl.DataFrame,
	nlp_features: pl.DataFrame,
	clean_reviews: pl.DataFrame,
) -> pl.DataFrame:
	"""Build daily I10-I11 feature_series rows.

	`clean_reviews` maps NLP review IDs to SKUs because `nlp_features` intentionally
	contains no SKU column.
	"""
	validate_daily_agg(daily_agg)
	validate(Table.NLP_FEATURES, nlp_features)
	validate(Table.CLEAN_REVIEWS, clean_reviews)
	if daily_agg.is_empty():
		return validate(Table.FEATURE_SERIES, empty_table(Table.FEATURE_SERIES))

	verified = daily_agg.select(
		"sku",
		"category",
		pl.col("date").alias("window_end"),
		pl.when(pl.col("n_all") > 0)
		.then(pl.col("n_verified_all").cast(pl.Float64) / pl.col("n_all"))
		.otherwise(None)
		.alias("verified_ratio_all"),
		pl.col("n_all").alias("verified_n"),
	)

	per_sku_mismatch = (
		clean_reviews.select("review_id", "sku", "category", "date")
		.join(nlp_features.select("review_id", "mismatch"), on="review_id", how="inner")
		.filter(pl.col("mismatch").is_not_null())
		.group_by("sku", "category", "date")
		.agg(
			pl.col("mismatch").mean().alias("mismatch_rate"),
			pl.len().alias("mismatch_n"),
		)
	)
	base = verified.join(
		per_sku_mismatch,
		left_on=["sku", "category", "window_end"],
		right_on=["sku", "category", "date"],
		how="left",
	)
	rows = pl.concat(
		[
			base.select(
				"sku",
				"category",
				"window_end",
				pl.lit("I10").alias("indicator_id"),
				pl.lit("mismatch_rate").alias("feature"),
				pl.col("mismatch_rate").cast(pl.Float64).alias("raw_value"),
				pl.col("mismatch_n").fill_null(0).cast(pl.Int32).alias("n_window"),
			),
			base.select(
				"sku",
				"category",
				"window_end",
				pl.lit("I11").alias("indicator_id"),
				pl.lit("verified_ratio_all").alias("feature"),
				pl.col("verified_ratio_all").cast(pl.Float64).alias("raw_value"),
				pl.col("verified_n").cast(pl.Int32).alias("n_window"),
			),
		],
		how="vertical",
	).sort(["sku", "window_end", "indicator_id"])
	return validate(Table.FEATURE_SERIES, conform(Table.FEATURE_SERIES, rows))
