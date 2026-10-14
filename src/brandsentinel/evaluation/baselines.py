"""Simple alert-severity policies for comparison on verified recall events."""

from __future__ import annotations

from typing import Any

import polars as pl

from brandsentinel.core.schemas import validate
from brandsentinel.core.types import Table
from brandsentinel.evaluation.metrics import evaluate_alerts


def evaluate_baselines(
	alerts: pl.DataFrame,
	recall_labels: pl.DataFrame,
	*,
	lookback_days: int = 90,
) -> dict[str, dict[str, Any]]:
	"""Compare severity policies and signal-family baselines against recall events.

	Signal-family baselines filter emitted alerts by their triggered indicators. They
	are not independently re-scored detectors; ``recall_labels`` contains positive
	events only and the alert table omits non-alert windows.
	"""
	validate(Table.ALERTS, alerts)
	validate(Table.RECALL_LABELS, recall_labels)

	baseline_indicators = {
		"rating_rolling": ["I3", "I4", "I5", "I6"],
		"volume_only": ["I1", "I2"],
		"keyword_only": ["I7", "I8"],
	}
	results = {
		"medium_or_high": evaluate_alerts(alerts, recall_labels, lookback_days=lookback_days),
		"high_only": evaluate_alerts(
			alerts.filter(pl.col("level") == "HIGH"),
			recall_labels,
			lookback_days=lookback_days,
		),
	}
	for name, indicators in baseline_indicators.items():
		baseline_alerts = alerts.filter(
			pl.col("triggered_ids").list.eval(pl.element().is_in(indicators)).list.any()
		)
		results[name] = evaluate_alerts(
			baseline_alerts, recall_labels, lookback_days=lookback_days
		)
	return results
