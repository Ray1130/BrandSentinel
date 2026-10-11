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
	"""Compare MEDIUM-or-HIGH alerts with a stricter HIGH-only policy.

	These are alert-policy comparisons, not precision baselines: ``recall_labels``
	contains positive events only and the alert table omits non-alert windows.
	"""
	validate(Table.ALERTS, alerts)
	validate(Table.RECALL_LABELS, recall_labels)

	return {
		"medium_or_high": evaluate_alerts(alerts, recall_labels, lookback_days=lookback_days),
		"high_only": evaluate_alerts(
			alerts.filter(pl.col("level") == "HIGH"),
			recall_labels,
			lookback_days=lookback_days,
		),
	}
