"""Aggregate clean_reviews into one daily row per SKU."""

from __future__ import annotations

import polars as pl

from brandsentinel.core.schemas import conform, empty_table
from brandsentinel.core.types import Table


def make_daily_agg(clean_reviews: pl.DataFrame) -> pl.DataFrame:
	"""Aggregate daily counts, retaining pre-filter counts for reliability indicators."""
	if clean_reviews.is_empty():
		return empty_table(Table.DAILY_AGG)

	keys = ["sku", "category", "date"]
	usable = clean_reviews.filter(~pl.col("is_spam")).group_by(keys).agg(
		pl.len().alias("n"),
		(pl.col("rating") <= 2).sum().alias("n_neg"),
		*[(pl.col("rating") == rating).sum().alias(f"n{rating}") for rating in range(1, 6)],
		pl.col("rating").cast(pl.Float64).sum().alias("sum_rating"),
		(pl.col("rating").cast(pl.Float64) ** 2).sum().alias("sumsq_rating"),
		pl.col("verified_purchase").fill_null(False).sum().alias("n_verified"),
	)
	all_reviews = clean_reviews.group_by(keys).agg(
		pl.len().alias("n_all"),
		pl.col("verified_purchase").fill_null(False).sum().alias("n_verified_all"),
	)
	start, end = clean_reviews["date"].min(), clean_reviews["date"].max()
	dates = pl.DataFrame({"date": pl.date_range(start, end, "1d", eager=True)})
	grid = clean_reviews.select("sku", "category").unique().join(dates, how="cross")
	result = (
		grid.join(usable, on=keys, how="left")
		.join(all_reviews, on=keys, how="left")
		.fill_null(0)
		.sort(["sku", "date"])
	)
	return conform(Table.DAILY_AGG, result)
