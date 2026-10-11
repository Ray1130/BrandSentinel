"""Match manually verified public recall records to Amazon review SKUs."""

from __future__ import annotations

import polars as pl

from brandsentinel.core.schemas import empty_table, validate
from brandsentinel.core.types import Table

_RECALL_COLUMNS = {
	"asin",
	"recall_id",
	"recall_date",
	"source",
	"source_url",
	"product_name",
	"manually_verified",
}


def match_recall_labels(
	recalls: pl.DataFrame,
	clean_reviews: pl.DataFrame,
	*,
	category: str,
) -> pl.DataFrame:
	"""Return verified exact-ASIN matches in the ``recall_labels`` contract.

	``recalls`` is a curated table from public recall records. Its ``asin`` must have
	been manually checked against ``source_url``; unreviewed records are not emitted.
	"""
	if not category:
		raise ValueError("category không được để trống")
	missing = sorted(_RECALL_COLUMNS - set(recalls.columns))
	if missing:
		raise ValueError(f"danh sách thu hồi thiếu cột bắt buộc: {missing}")
	if recalls.schema["manually_verified"] != pl.Boolean:
		raise ValueError("manually_verified phải có kiểu Boolean")

	validate(Table.CLEAN_REVIEWS, clean_reviews)
	if category not in clean_reviews["category"].unique().to_list():
		raise ValueError(f"không tìm thấy clean_reviews cho category={category!r}")

	review_skus = (
		clean_reviews.filter(pl.col("category") == category)
		.select(
			pl.col("sku").alias("sku"),
			pl.col("sku").str.strip_chars().str.to_uppercase().alias("_asin"),
		)
		.unique(subset=["_asin"], keep="first")
	)
	verified_recalls = (
		recalls.filter(pl.col("manually_verified"))
		.with_columns(
			pl.col("asin").cast(pl.String).str.strip_chars().str.to_uppercase().alias("_asin")
		)
		.filter(pl.col("_asin").is_not_null() & (pl.col("_asin") != ""))
	)
	if verified_recalls.is_empty() or review_skus.is_empty():
		return empty_table(Table.RECALL_LABELS)

	matches = verified_recalls.join(review_skus, on="_asin", how="inner").select(
		"sku",
		pl.lit(category).alias("category"),
		"recall_id",
		"recall_date",
		"source",
		"source_url",
		"product_name",
		pl.lit("exact_asin").alias("match_type"),
		pl.lit(1.0).alias("match_confidence"),
	)
	return validate(Table.RECALL_LABELS, matches)
