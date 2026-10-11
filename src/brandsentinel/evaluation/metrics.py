"""Event-level evaluation of alerts against verified product recall labels."""

from __future__ import annotations

from statistics import mean, median
from typing import Any

import polars as pl

from brandsentinel.core.schemas import validate
from brandsentinel.core.types import Table

_EVENT_KEY = ["sku", "category", "recall_id", "recall_date"]


def match_alerts_to_recalls(
	alerts: pl.DataFrame,
	recall_labels: pl.DataFrame,
	*,
	lookback_days: int = 90,
) -> pl.DataFrame:
	"""Return one outcome per recall event, matching alerts before its publication date.

	An event is detected when an alert for the same SKU and category occurs in the
	inclusive ``[recall_date - lookback_days, recall_date]`` interval. The earliest
	alert in that interval determines lead time. Missing recall labels are not treated
	as negative examples.
	"""
	if lookback_days < 0:
		raise ValueError("lookback_days phải >= 0")

	validate(Table.ALERTS, alerts)
	validate(Table.RECALL_LABELS, recall_labels)

	matched = (
		recall_labels.join(
			alerts.select("sku", "category", "window_end"),
			on=["sku", "category"],
			how="inner",
		)
		.filter(
			pl.col("window_end").is_between(
				pl.col("recall_date").dt.offset_by(f"-{lookback_days}d"),
				pl.col("recall_date"),
				closed="both",
			)
		)
		.group_by(_EVENT_KEY)
		.agg(
			pl.col("window_end").min().alias("first_alert_date"),
			pl.len().cast(pl.Int32).alias("alert_count"),
		)
	)

	return (
		recall_labels.join(matched, on=_EVENT_KEY, how="left")
		.with_columns(
			pl.col("alert_count").fill_null(0),
			pl.col("first_alert_date").is_not_null().alias("detected"),
			(
				(pl.col("recall_date") - pl.col("first_alert_date"))
				.dt.total_days()
				.cast(pl.Int32)
				.alias("lead_time_days")
			),
		)
		.sort("recall_date", "sku", "recall_id")
	)


def evaluate_alerts(
	alerts: pl.DataFrame,
	recall_labels: pl.DataFrame,
	*,
	lookback_days: int = 90,
) -> dict[str, Any]:
	"""Summarize event recall and warning lead time; precision is not identifiable."""
	events = match_alerts_to_recalls(alerts, recall_labels, lookback_days=lookback_days)
	detected = events.filter(pl.col("detected"))
	lead_times = detected["lead_time_days"].drop_nulls().to_list()
	event_count = events.height
	detected_count = detected.height

	return {
		"lookback_days": lookback_days,
		"recall_events": event_count,
		"detected_events": detected_count,
		"missed_events": event_count - detected_count,
		"event_recall": detected_count / event_count if event_count else None,
		"mean_lead_time_days": mean(lead_times) if lead_times else None,
		"median_lead_time_days": median(lead_times) if lead_times else None,
		"alerts_in_window": int(events["alert_count"].sum() or 0),
	}
