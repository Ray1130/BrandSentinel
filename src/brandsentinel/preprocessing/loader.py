"""Load Amazon Reviews 2023 JSONL into the clean_reviews contract."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import polars as pl

from brandsentinel.core.ids import review_id
from brandsentinel.core.schemas import conform, empty_table
from brandsentinel.core.types import Table
from brandsentinel.preprocessing.validate import validate_clean_reviews


def _read_source(source: str | Path | Iterable[str | Path] | pl.DataFrame) -> pl.DataFrame:
	if isinstance(source, pl.DataFrame):
		return source
	if isinstance(source, (str, Path)):
		path = Path(source)
		files = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
	else:
		files = [Path(path) for path in source]
	if not files:
		raise FileNotFoundError(f"không tìm thấy file JSONL: {source}")
	frames = [pl.read_ndjson(path) for path in files]
	return pl.concat(frames, how="diagonal_relaxed")


def load_reviews(
	source: str | Path | Iterable[str | Path] | pl.DataFrame,
	category: str,
	*,
	skus: list[str] | None = None,
) -> pl.DataFrame:
	"""Map source records to clean_reviews; timestamps are Unix milliseconds in UTC.

	`user_id` and `verified_purchase` may be absent in source data and become null.
	"""
	if not category:
		raise ValueError("category không được để trống")
	raw = _read_source(source)
	required = ("parent_asin", "timestamp", "rating")
	missing = [column for column in required if column not in raw.columns]
	if missing:
		raise ValueError(f"Amazon JSONL thiếu cột bắt buộc: {missing}")
	if raw.is_empty():
		return empty_table(Table.CLEAN_REVIEWS)

	raw = raw.with_columns(
		pl.col("parent_asin").cast(pl.String).alias("sku"),
		pl.col("timestamp").cast(pl.Int64, strict=False).alias("_timestamp_ms"),
		pl.col("user_id").cast(pl.String, strict=False)
		if "user_id" in raw.columns
		else pl.lit(None, dtype=pl.String).alias("user_id"),
		pl.col("text").cast(pl.String, strict=False).alias("text_raw")
		if "text" in raw.columns
		else pl.lit(None, dtype=pl.String).alias("text_raw"),
		pl.col("verified_purchase").cast(pl.Boolean, strict=False)
		if "verified_purchase" in raw.columns
		else pl.lit(None, dtype=pl.Boolean).alias("verified_purchase"),
		pl.col("rating").cast(pl.Int8, strict=False).alias("rating"),
	).with_columns(
		pl.from_epoch(pl.col("_timestamp_ms"), time_unit="ms")
		.dt.replace_time_zone("UTC")
		.alias("ts")
	)

	if raw["_timestamp_ms"].null_count():
		raise ValueError("timestamp phải là Unix milliseconds hợp lệ")
	ids = [
		review_id(sku, user_id, timestamp_ms)
		for sku, user_id, timestamp_ms in raw.select(
			"sku", "user_id", "_timestamp_ms"
		).iter_rows()
	]
	clean = raw.with_columns(
		pl.Series("review_id", ids, dtype=pl.String),
		pl.lit(category, dtype=pl.String).alias("category"),
		pl.col("ts").dt.date().alias("date"),
		pl.lit(False).alias("is_spam"),
		pl.lit(None, dtype=pl.String).alias("spam_reason"),
		pl.lit(None, dtype=pl.String).alias("text_norm"),
	).select(
		"review_id", "sku", "category", "user_id", "ts", "date", "rating", "text_raw",
		"text_norm", "verified_purchase", "is_spam", "spam_reason",
	)
	clean = conform(Table.CLEAN_REVIEWS, clean)
	if skus is not None:
		clean = clean.filter(pl.col("sku").is_in(skus))
	return validate_clean_reviews(clean)
