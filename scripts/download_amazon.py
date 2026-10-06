"""Stream a category of Amazon Reviews 2023 and select sufficiently mature SKUs."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from collections import Counter
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from brandsentinel.core.config import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
_SAFE_CATEGORY = re.compile(r"[A-Za-z0-9_.-]+")


def _timestamp_date(timestamp: Any) -> date:
	return datetime.fromtimestamp(int(timestamp) / 1000, UTC).date()


def _accumulate(stats: dict[str, dict[str, Any]], record: dict[str, Any]) -> None:
	sku = record.get("parent_asin")
	timestamp = record.get("timestamp")
	if not sku or timestamp is None:
		return
	review_date = _timestamp_date(timestamp)
	week = review_date.toordinal() // 7
	item = stats.setdefault(
		str(sku), {"first": review_date, "last": review_date, "weeks": Counter(), "count": 0}
	)
	item["first"] = min(item["first"], review_date)
	item["last"] = max(item["last"], review_date)
	item["weeks"][week] += 1
	item["count"] += 1


def _select_skus(
	stats: dict[str, dict[str, Any]],
	*,
	min_history_days: int,
	min_total_reviews: int,
	min_median_weekly_reviews: float,
) -> dict[str, dict[str, Any]]:
	selected: dict[str, dict[str, Any]] = {}
	for sku, item in stats.items():
		history_days = (item["last"] - item["first"]).days + 1
		first_week = item["first"].toordinal() // 7
		last_week = item["last"].toordinal() // 7
		weekly_counts = [item["weeks"].get(week, 0) for week in range(first_week, last_week + 1)]
		median_weekly = statistics.median(weekly_counts)
		if (
			history_days >= min_history_days
			and item["count"] >= min_total_reviews
			and median_weekly >= min_median_weekly_reviews
		):
			selected[sku] = {
				"sku": sku,
				"history_start": item["first"].isoformat(),
				"history_end": item["last"].isoformat(),
				"history_days": history_days,
				"review_count": item["count"],
				"median_weekly_reviews": median_weekly,
			}
	return selected


def _parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--category", required=True, help="Amazon category, e.g. Baby_Products")
	parser.add_argument(
		"--output", type=Path, help="JSONL destination (default: data/raw/<category>.jsonl)"
	)
	parser.add_argument(
		"--sku-output", type=Path, help="SKU CSV destination (default: data/interim/<category>_skus.csv)"
	)
	parser.add_argument("--min-history-days", type=int)
	parser.add_argument("--min-total-reviews", type=int)
	parser.add_argument("--min-median-weekly-reviews", type=float)
	return parser


def main(argv: list[str] | None = None) -> int:
	args = _parser().parse_args(argv)
	if not _SAFE_CATEGORY.fullmatch(args.category):
		raise SystemExit("category chỉ được chứa A-Z, a-z, 0-9, dấu chấm, gạch dưới hoặc gạch ngang")
	try:
		from datasets import load_dataset
	except ImportError as exc:
		raise SystemExit(
			'Chạy bằng `uv run --with "datasets>=2.18,<4" python scripts/download_amazon.py ...`.'
		) from exc

	cfg = load_config(REPO_ROOT / "configs").default.data.sku_selection
	min_history = (
		args.min_history_days if args.min_history_days is not None else cfg.min_history_days
	)
	min_reviews = (
		args.min_total_reviews
		if args.min_total_reviews is not None
		else cfg.min_total_reviews
	)
	min_weekly = (
		args.min_median_weekly_reviews
		if args.min_median_weekly_reviews is not None
		else cfg.min_median_weekly_reviews
	)

	def stream():
		return load_dataset(
			"McAuley-Lab/Amazon-Reviews-2023",
			f"raw_review_{args.category}",
			split="full",
			streaming=True,
			trust_remote_code=True,
		)

	stats: dict[str, dict[str, Any]] = {}
	missing_fields: Counter[str] = Counter()
	row_count = 0
	for record in stream():
		row_count += 1
		for field in ("user_id", "verified_purchase"):
			if record.get(field) is None:
				missing_fields[field] += 1
		_accumulate(stats, record)

	selected = _select_skus(
		stats,
		min_history_days=min_history,
		min_total_reviews=min_reviews,
		min_median_weekly_reviews=min_weekly,
	)
	output = args.output or REPO_ROOT / "data" / "raw" / f"{args.category}.jsonl"
	sku_output = args.sku_output or REPO_ROOT / "data" / "interim" / f"{args.category}_skus.csv"
	report_output = sku_output.with_suffix(".report.json")
	output.parent.mkdir(parents=True, exist_ok=True)
	sku_output.parent.mkdir(parents=True, exist_ok=True)

	temp_output = output.with_name(output.name + ".tmp")
	written = 0
	with temp_output.open("w", encoding="utf-8") as target:
		for record in stream():
			if str(record.get("parent_asin")) in selected:
				target.write(json.dumps(record, ensure_ascii=False) + "\n")
				written += 1
	temp_output.replace(output)

	columns = [
		"sku",
		"history_start",
		"history_end",
		"history_days",
		"review_count",
		"median_weekly_reviews",
	]
	with sku_output.open("w", encoding="utf-8", newline="") as target:
		writer = csv.DictWriter(target, fieldnames=columns)
		writer.writeheader()
		writer.writerows(selected.values())
	report_output.write_text(
		json.dumps(
			{
				"category": args.category,
				"source_rows": row_count,
				"selected_skus": len(selected),
				"exported_reviews": written,
				"missing_or_null_fields": dict(missing_fields),
				"thresholds": {
					"min_history_days": min_history,
					"min_total_reviews": min_reviews,
					"min_median_weekly_reviews": min_weekly,
				},
			},
			ensure_ascii=False,
			indent=2,
		)
		+ "\n",
		encoding="utf-8",
	)
	print(f"Category: {args.category}; source reviews: {row_count}")
	print(f"Selected {len(selected)} SKUs; exported {written} reviews")
	print(f"SKU list: {sku_output}\nRaw JSONL: {output}\nReport: {report_output}")
	for field in ("user_id", "verified_purchase"):
		absent = missing_fields[field]
		if row_count:
			print(f"{field}: {absent} missing/null ({absent / row_count:.2%})")
		else:
			print(f"{field}: no source rows")
	return 0


if __name__ == "__main__":
	sys.exit(main())
