from datetime import date

import polars as pl
import pytest

from brandsentinel.core.schemas import empty_table
from brandsentinel.core.types import Table
from brandsentinel.evaluation.baselines import evaluate_baselines
from brandsentinel.evaluation.metrics import evaluate_alerts, match_alerts_to_recalls


def _tables():
	alerts = pl.DataFrame(
		[
			("SKU-1", "category", date(2025, 1, 1), 0.7, "MEDIUM", ["content"], 1, False, ["I8"]),
			("SKU-1", "category", date(2025, 1, 8), 0.9, "HIGH", ["content"], 2, False, ["I8"]),
			("SKU-1", "category", date(2025, 1, 12), 0.8, "HIGH", ["content"], 1, False, ["I8"]),
			("SKU-2", "other", date(2025, 1, 8), 0.9, "HIGH", ["content"], 1, False, ["I8"]),
		],
		schema=empty_table(Table.ALERTS).schema,
		orient="row",
	)
	labels = pl.DataFrame(
		[
			(
				"SKU-1",
				"category",
				"R-1",
				date(2025, 1, 10),
				"CPSC",
				"https://example.com/recall",
				"Product",
				"exact_asin",
				1.0,
			),
			(
				"SKU-2",
				"category",
				"R-2",
				date(2025, 1, 10),
				"CPSC",
				"https://example.com/recall-2",
				"Product",
				"exact_asin",
				1.0,
			),
		],
		schema=empty_table(Table.RECALL_LABELS).schema,
		orient="row",
	)
	return alerts, labels


def test_matches_pre_recall_alerts_inclusive_and_excludes_later_alerts():
	alerts, labels = _tables()
	events = match_alerts_to_recalls(alerts, labels, lookback_days=9)

	assert events.select("recall_id", "detected", "first_alert_date", "lead_time_days", "alert_count").rows() == [
		("R-1", True, date(2025, 1, 1), 9, 2),
		("R-2", False, None, None, 0),
	]


def test_metrics_are_event_recall_not_alert_precision():
	alerts, labels = _tables()
	metrics = evaluate_alerts(alerts, labels, lookback_days=9)

	assert metrics == {
		"lookback_days": 9,
		"recall_events": 2,
		"detected_events": 1,
		"missed_events": 1,
		"event_recall": 0.5,
		"mean_lead_time_days": 9.0,
		"median_lead_time_days": 9,
		"alerts_in_window": 2,
	}


def test_baselines_compare_all_severities_with_high_only():
	alerts, labels = _tables()
	results = evaluate_baselines(alerts, labels, lookback_days=9)

	assert results["medium_or_high"]["event_recall"] == 0.5
	assert results["high_only"]["event_recall"] == 0.5
	assert results["high_only"]["mean_lead_time_days"] == 2.0
	assert results["high_only"]["alerts_in_window"] == 1


def test_empty_recall_labels_have_undefined_recall():
	alerts, _ = _tables()
	metrics = evaluate_alerts(alerts, empty_table(Table.RECALL_LABELS))

	assert metrics["recall_events"] == 0
	assert metrics["event_recall"] is None
	assert metrics["mean_lead_time_days"] is None


def test_negative_lookback_is_rejected():
	alerts, labels = _tables()
	with pytest.raises(ValueError, match="lookback_days"):
		evaluate_alerts(alerts, labels, lookback_days=-1)