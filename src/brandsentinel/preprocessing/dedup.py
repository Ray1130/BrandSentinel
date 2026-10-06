"""Exact review deduplication."""

from __future__ import annotations

import polars as pl

from brandsentinel.core.logging import AuditLog

EXACT_KEY = ("sku", "user_id", "text_norm")


def deduplicate_reviews(
	df: pl.DataFrame,
	*,
	audit: AuditLog | None = None,
	keep: str = "earliest",
) -> pl.DataFrame:
	"""Drop exact duplicates by SKU, reviewer, and normalized text."""
	if keep not in {"earliest", "latest"}:
		raise ValueError("keep phải là 'earliest' hoặc 'latest'")
	missing = [column for column in (*EXACT_KEY, "ts") if column not in df.columns]
	if missing:
		raise ValueError(f"clean_reviews thiếu cột dedup: {missing}")
	ordered = df.sort("ts", maintain_order=True, descending=keep == "latest")
	result = ordered.unique(subset=list(EXACT_KEY), keep="first", maintain_order=True)
	if audit is not None:
		audit.record("deduplicate_exact", df.height, result.height, key=list(EXACT_KEY))
	return result
