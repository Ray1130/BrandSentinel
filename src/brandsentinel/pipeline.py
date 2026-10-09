"""Pipeline stage entry points."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl

from brandsentinel.core.config import Config, get_config
from brandsentinel.core.io import write_table
from brandsentinel.core.logging import AuditLog
from brandsentinel.core.types import Table
from brandsentinel.preprocessing.aggregate import make_daily_agg
from brandsentinel.preprocessing.dedup import deduplicate_reviews
from brandsentinel.preprocessing.loader import load_reviews
from brandsentinel.preprocessing.normalize import normalize_reviews
from brandsentinel.preprocessing.spam import filter_eligible_skus, flag_spam_reviews


def run_preprocess(
	category: str,
	*,
	start: date | None = None,
	end: date | None = None,
	source: str | Path | pl.DataFrame | None = None,
	config: Config | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame, Path | None]:
	"""Preprocess one category, persist its tables, and optionally write an audit log."""
	cfg = config or get_config()
	if start and end and start > end:
		raise ValueError("start phải trước hoặc bằng end")
	input_source = source if source is not None else cfg.path("data_raw") / f"{category}.jsonl"
	audit = AuditLog("preprocess") if cfg.default.logging.audit_log else None
	clean = load_reviews(input_source, category)
	clean = normalize_reviews(
		clean,
		audit=audit,
		unicode_form=cfg.default.preprocessing.normalize.unicode_form,
		lowercase=cfg.default.preprocessing.normalize.lowercase,
		strip_html=cfg.default.preprocessing.normalize.strip_html,
		collapse_repeated_chars=cfg.default.preprocessing.normalize.collapse_repeated_chars,
		emoji_to_token=cfg.default.preprocessing.normalize.emoji_to_token,
	)
	clean = deduplicate_reviews(
		clean,
		audit=audit,
		keep=cfg.default.preprocessing.dedup.keep,
	)
	clean = flag_spam_reviews(clean, config=cfg.default.preprocessing.spam, audit=audit)
	daily = make_daily_agg(clean)
	daily = filter_eligible_skus(
		daily,
		selection=cfg.default.data.sku_selection,
		audit=audit,
	)
	selected_skus = daily["sku"].unique().to_list() if not daily.is_empty() else []
	clean = clean.filter(pl.col("sku").is_in(selected_skus))
	if start is not None:
		clean = clean.filter(pl.col("date") >= start)
		daily = daily.filter(pl.col("date") >= start)
	if end is not None:
		clean = clean.filter(pl.col("date") <= end)
		daily = daily.filter(pl.col("date") <= end)
	write_table(clean, Table.CLEAN_REVIEWS, cfg)
	write_table(daily, Table.DAILY_AGG, cfg)
	audit_path = audit.write(cfg.path("data_interim") / "audit") if audit is not None else None
	return clean, daily, audit_path
