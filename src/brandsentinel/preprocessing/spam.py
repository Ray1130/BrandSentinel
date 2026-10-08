"""Spam flagging and configured SKU eligibility for preprocessing."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from brandsentinel.core.logging import AuditLog
from brandsentinel.preprocessing.validate import validate_daily_agg

if TYPE_CHECKING:
	from brandsentinel.core.config import SkuSelectionCfg, SpamCfg


def flag_spam_reviews(
	df: pl.DataFrame,
	*,
	config: SpamCfg,
	audit: AuditLog | None = None,
) -> pl.DataFrame:
	"""Set `is_spam` and explain each matched rule without dropping reviews."""
	if df.is_empty():
		return df
	required = {"sku", "date", "ts", "user_id", "text_norm"}
	missing = sorted(required - set(df.columns))
	if missing:
		raise ValueError(f"clean_reviews thiếu cột gắn cờ spam: {missing}")

	user_counts = (
		df.filter(pl.col("user_id").is_not_null())
		.group_by("user_id", "date")
		.len()
		.rename({"len": "_user_day_count"})
	)
	template_counts = (
		df.filter(pl.col("text_norm").is_not_null() & (pl.col("text_norm") != ""))
		.group_by("sku", "date", "text_norm")
		.len()
		.rename({"len": "_template_count"})
	)
	second_counts = (
		df.group_by("sku", "ts")
		.len()
		.rename({"len": "_same_second_count"})
	)
	work = (
		df.join(user_counts, on=["user_id", "date"], how="left")
		.join(template_counts, on=["sku", "date", "text_norm"], how="left")
		.join(second_counts, on=["sku", "ts"], how="left")
		.with_columns(
			pl.col("text_norm").str.count_matches(r"\S+").fill_null(0).alias("_token_count")
		)
	)
	rules = [
		("too_short", pl.col("_token_count") < config.min_tokens),
		(
			"user_day_burst",
			pl.col("_user_day_count").fill_null(0) > config.max_reviews_per_user_per_day,
		),
		(
			"repeated_template",
			pl.col("_template_count").fill_null(0) >= config.template_repeat_min_count,
		),
		(
			"same_second_burst",
			pl.col("_same_second_count").fill_null(0) >= config.same_second_burst_min,
		),
	]
	flag_exprs = [condition.alias(f"_flag_{name}") for name, condition in rules]
	work = work.with_columns(flag_exprs).with_columns(
		pl.concat_list(
			[
				pl.when(pl.col(f"_flag_{name}")).then(pl.lit(name)).otherwise(None)
				for name, _ in rules
			]
		)
		.list.drop_nulls()
		.list.join("|")
		.alias("_spam_reasons")
	)
	flagged_by_rule = {
		f"{name}_flagged": int(work[f"_flag_{name}"].sum() or 0) for name, _ in rules
	}
	result = work.with_columns(
		(pl.col("_spam_reasons") != "").alias("is_spam"),
		pl.when(pl.col("_spam_reasons") != "")
		.then(pl.col("_spam_reasons"))
		.otherwise(None)
		.alias("spam_reason"),
	).drop(
		"_user_day_count",
		"_template_count",
		"_same_second_count",
		"_token_count",
		"_spam_reasons",
		*[f"_flag_{name}" for name, _ in rules],
	)
	if audit is not None:
		audit.record(
			"flag_spam",
			df.height,
			result.height,
			flagged=int(result["is_spam"].sum()),
			**flagged_by_rule,
		)
	return result


def filter_eligible_skus(
	daily_agg: pl.DataFrame,
	*,
	selection: SkuSelectionCfg,
	audit: AuditLog | None = None,
) -> pl.DataFrame:
	"""Keep daily rows only for SKUs meeting all configured history thresholds."""
	validate_daily_agg(daily_agg)
	if daily_agg.is_empty():
		return daily_agg
	required = {"sku", "date", "n_all"}
	missing = sorted(required - set(daily_agg.columns))
	if missing:
		raise ValueError(f"daily_agg thiếu cột chọn SKU: {missing}")

	observed = daily_agg.filter(pl.col("n_all") > 0).with_columns(
		((pl.col("date").cast(pl.Int32) + 719_163) // 7).alias("_week_key")
	)
	weekly = observed.group_by("sku", "_week_key").agg(
		pl.col("n_all").sum().alias("_weekly_count")
	)
	stats = observed.group_by("sku").agg(
		pl.col("date").min().alias("_first_date"),
		pl.col("date").max().alias("_last_date"),
		pl.col("n_all").sum().alias("_review_count"),
		pl.col("_week_key").min().alias("_first_week"),
		pl.col("_week_key").max().alias("_last_week"),
	)
	calendar_weeks = stats.select(
		"sku",
		pl.int_ranges(pl.col("_first_week"), pl.col("_last_week") + 1).alias("_week_key"),
	).explode("_week_key")
	weekly_stats = (
		calendar_weeks.join(weekly, on=["sku", "_week_key"], how="left")
		.with_columns(pl.col("_weekly_count").fill_null(0))
		.group_by("sku")
		.agg(pl.col("_weekly_count").median().alias("_median_weekly_reviews"))
	)
	stats = stats.join(weekly_stats, on="sku", how="left").with_columns(
		(pl.col("_last_date") - pl.col("_first_date")).dt.total_days().add(1).alias("_history_days")
	)
	eligible = stats.filter(
		(pl.col("_history_days") >= selection.min_history_days)
		& (pl.col("_review_count") >= selection.min_total_reviews)
		& (pl.col("_median_weekly_reviews") >= selection.min_median_weekly_reviews)
	)["sku"].to_list()

	result = daily_agg.filter(pl.col("sku").is_in(eligible))
	if audit is not None:
		audit.record(
			"select_eligible_skus",
			daily_agg.height,
			result.height,
			skus_in=daily_agg["sku"].n_unique(),
			skus_out=len(eligible),
			excluded_skus=sorted(set(daily_agg["sku"].to_list()) - set(eligible)),
			thresholds={
				"min_history_days": selection.min_history_days,
				"min_total_reviews": selection.min_total_reviews,
				"min_median_weekly_reviews": selection.min_median_weekly_reviews,
			},
		)
	return result
