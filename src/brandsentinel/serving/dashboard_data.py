"""Read-only data access and presentation helpers for the Streamlit dashboard."""

from __future__ import annotations

from collections import defaultdict
from datetime import date, timedelta
from functools import lru_cache
from typing import Any

import polars as pl

from brandsentinel.core.config import Config, load_config
from brandsentinel.core.io import read_json, read_table
from brandsentinel.core.schemas import empty_table
from brandsentinel.core.types import INDICATOR_IDS, Table
from brandsentinel.llm.evidence import build_evidence_pack
from brandsentinel.llm.schemas import EvidencePack, RecommendationOutput
from brandsentinel.scoring.weights import calculate_weights

_GROUPS = ("volume", "rating", "content")


def available_categories(config: Config) -> list[str]:
    """Return category partitions available in alerts, falling back to daily aggregates."""
    categories: set[str] = set()
    for table in (Table.ALERTS, Table.DAILY_AGG):
        base = (
            config.path("data_processed" if table == Table.ALERTS else "data_interim") / table.value
        )
        if base.is_dir():
            categories.update(
                p.name.removeprefix("category=") for p in base.glob("category=*") if p.is_dir()
            )
        if categories:
            break
    return sorted(categories)


def _partition_fingerprint(
    config: Config, table: Table, category: str
) -> tuple[tuple[str, int, int], ...]:
    zone = (
        "data_interim"
        if table
        in {
            Table.CLEAN_REVIEWS,
            Table.DAILY_AGG,
            Table.NLP_FEATURES,
            Table.RECALL_LABELS,
        }
        else "data_processed"
    )
    directory = config.path(zone) / table.value / f"category={category}"
    if not directory.is_dir():
        return ()
    return tuple(
        (str(path), path.stat().st_mtime_ns, path.stat().st_size)
        for path in sorted(directory.glob("*.parquet"))
    )


@lru_cache(maxsize=64)
def _cached_read(
    root: str,
    config_dir: str,
    table_value: str,
    category: str,
    start: date | None,
    end: date | None,
    skus: tuple[str, ...] | None,
    review_ids: tuple[str, ...] | None,
    fingerprint: tuple[tuple[str, int, int], ...],
) -> pl.DataFrame:
    del fingerprint
    config = load_config(config_dir)
    return read_table(
        Table(table_value),
        config,
        category=category,
        start=start,
        end=end,
        skus=list(skus) if skus is not None else None,
        review_ids=list(review_ids) if review_ids is not None else None,
    )


def read_category_table(
    table: Table,
    config: Config,
    category: str,
    *,
    start: date | None = None,
    end: date | None = None,
    skus: list[str] | None = None,
    review_ids: list[str] | None = None,
) -> pl.DataFrame:
    """Read one category partition with an mtime-aware cache and pushed-down filters."""
    return _cached_read(
        str(config.root),
        str(config.path("configs")),
        table.value,
        category,
        start,
        end,
        tuple(sorted(skus)) if skus is not None else None,
        tuple(sorted(review_ids)) if review_ids is not None else None,
        _partition_fingerprint(config, table, category),
    )


def load_alerts(
    config: Config,
    category: str,
    *,
    start: date | None = None,
    end: date | None = None,
    skus: list[str] | None = None,
) -> pl.DataFrame:
    """Load known MEDIUM/HIGH alerts for a category and optional inclusive date range."""
    try:
        return read_category_table(Table.ALERTS, config, category, start=start, end=end, skus=skus)
    except FileNotFoundError:
        return empty_table(Table.ALERTS)


def snapshot(alerts: pl.DataFrame, as_of: date) -> pl.DataFrame:
    """Return each SKU's latest recorded alert at or before the selected date."""
    if alerts.is_empty():
        return alerts
    return (
        alerts.filter(pl.col("window_end") <= as_of)
        .sort(["sku", "window_end"])
        .unique(subset="sku", keep="last", maintain_order=True)
        .sort(["level", "score"], descending=True)
    )


def _level_since(alerts: pl.DataFrame, selected: dict[str, Any], step_days: int) -> date:
    history = (
        alerts.filter(
            (pl.col("sku") == selected["sku"]) & (pl.col("window_end") <= selected["window_end"])
        )
        .sort("window_end", descending=True)
        .to_dicts()
    )
    current_day = selected["window_end"]
    current_level = selected["level"]
    for row in history[1:]:
        if row["level"] != current_level or (current_day - row["window_end"]).days != step_days:
            break
        current_day = row["window_end"]
    return current_day


def priority_table(
    alerts: pl.DataFrame, as_of: date, limit: int = 200, *, step_days: int = 1
) -> pl.DataFrame:
    """Build the ranked latest-per-SKU table, including the known-level start date."""
    if limit < 1:
        raise ValueError("limit phải lớn hơn 0")
    rows = snapshot(alerts, as_of).to_dicts()
    order = {"HIGH": 2, "MEDIUM": 1, "LOW": 0}
    rows.sort(
        key=lambda row: (
            order.get(str(row["level"]), -1),
            float(row["score"]),
            "I8" in (row.get("triggered_ids") or []),
        ),
        reverse=True,
    )
    for row in rows:
        row["level_since"] = _level_since(alerts, row, step_days)
        row["active_indicator_count"] = len(
            [i for i in row.get("triggered_ids", []) if i in INDICATOR_IDS[:9]]
        )
        row["active_groups"] = row.get("groups_active") or []
    return pl.DataFrame(rows[:limit]) if rows else pl.DataFrame()


def kpis(alerts: pl.DataFrame, as_of: date) -> dict[str, int | None]:
    """Return snapshot KPI counts; coverage is unavailable in the current table contract."""
    current = snapshot(alerts, as_of)
    high_skus = set(current.filter(pl.col("level") == "HIGH")["sku"].to_list())
    recent = alerts.filter(
        (pl.col("window_end") <= as_of)
        & (pl.col("window_end") >= as_of - timedelta(days=6))
        & (pl.col("level") == "HIGH")
    ).sort("window_end")
    newly_high: set[str] = set()
    for sku in high_skus:
        rows = recent.filter(pl.col("sku") == sku)
        if rows.is_empty():
            continue
        earliest = rows["window_end"].min()
        previous = (
            alerts.filter((pl.col("sku") == sku) & (pl.col("window_end") < earliest))
            .sort("window_end", descending=True)
            .head(1)
        )
        if previous.is_empty() or previous["level"][0] != "HIGH":
            newly_high.add(sku)
    return {
        "high": len(high_skus),
        "medium": current.filter(pl.col("level") == "MEDIUM")["sku"].n_unique()
        if not current.is_empty()
        else 0,
        "verify_flag": current.filter(pl.col("verify_flag"))["sku"].n_unique()
        if not current.is_empty()
        else 0,
        "new_high_7d": len(newly_high),
        "scored_skus": None,
        "insufficient_data_skus": None,
    }


def overview_metrics(
    clean_reviews: pl.DataFrame,
    nlp_features: pl.DataFrame,
    alerts: pl.DataFrame,
    as_of: date,
) -> tuple[list[dict[str, Any]], dict[str, int | float | None]]:
    """Build a product snapshot and category KPIs from the latest 7/14 review days."""
    current_start = as_of - timedelta(days=6)
    previous_start = as_of - timedelta(days=13)
    valid = (
        clean_reviews.filter(
            (pl.col("date") >= previous_start)
            & (pl.col("date") <= as_of)
            & ~pl.col("is_spam").fill_null(False)
        )
        if not clean_reviews.is_empty()
        else clean_reviews
    )
    sentiment_by_id: dict[str, float | None] = {}
    if not nlp_features.is_empty():
        sentiment_by_id = dict(nlp_features.select("review_id", "sentiment_score").iter_rows())

    weekly: dict[str, dict[str, dict[str, float]]] = defaultdict(
        lambda: {
            "current": defaultdict(float),
            "previous": defaultdict(float),
        }
    )
    for review in valid.to_dicts():
        sku = str(review["sku"])
        period = "current" if review["date"] >= current_start else "previous"
        bucket = weekly[sku][period]
        bucket["count"] += 1
        bucket["rating_total"] += float(review["rating"])
        bucket["low_rating_count"] += int(review["rating"] <= 2)
        sentiment = sentiment_by_id.get(str(review["review_id"]))
        if sentiment is not None:
            bucket["sentiment_count"] += 1
            bucket["negative_sentiment_count"] += int(sentiment < 0)

    current_alerts = snapshot(alerts, as_of)
    alert_by_sku = {str(row["sku"]): row for row in current_alerts.to_dicts()}
    all_skus = set(weekly) | set(alert_by_sku)
    rows: list[dict[str, Any]] = []
    rank = {"HIGH": 2, "MEDIUM": 1}

    def sentiment_rate(bucket: dict[str, float]) -> float | None:
        denominator = bucket["sentiment_count"]
        return bucket["negative_sentiment_count"] / denominator if denominator else None

    for sku in all_skus:
        current = weekly[sku]["current"]
        previous = weekly[sku]["previous"]
        current_sentiment = sentiment_rate(current)
        previous_sentiment = sentiment_rate(previous)
        if current_sentiment is None or previous_sentiment is None:
            trend = None
        else:
            trend = current_sentiment - previous_sentiment

        alert = alert_by_sku.get(sku)
        rows.append(
            {
                "sku": sku,
                "avg_rating": (
                    current["rating_total"] / current["count"] if current["count"] else None
                ),
                "low_rating_pct": (
                    current["low_rating_count"] / current["count"] if current["count"] else None
                ),
                "negative_sentiment_pct": current_sentiment,
                "negative_trend": trend,
                "level": alert["level"] if alert else None,
                "score": alert["score"] if alert else None,
                "window_end": alert["window_end"] if alert else None,
                "alert": alert,
            }
        )

    rows.sort(
        key=lambda row: (
            rank.get(str(row["level"]), 0),
            float(row["score"] or 0.0),
            row["sku"],
        ),
        reverse=True,
    )
    current_reviews = sum(bucket["current"]["count"] for bucket in weekly.values())
    current_sentiment_count = sum(
        bucket["current"]["sentiment_count"] for bucket in weekly.values()
    )
    total_negative_sentiment = sum(
        bucket["current"]["negative_sentiment_count"] for bucket in weekly.values()
    )
    metrics = {
        "valid_reviews": int(current_reviews),
        "high_risk": current_alerts.filter(pl.col("level") == "HIGH")["sku"].n_unique()
        if not current_alerts.is_empty()
        else 0,
        "medium_risk": current_alerts.filter(pl.col("level") == "MEDIUM")["sku"].n_unique()
        if not current_alerts.is_empty()
        else 0,
        "negative_sentiment_pct": (
            total_negative_sentiment / current_sentiment_count if current_sentiment_count else None
        ),
        "window_start": current_start,
    }
    return rows, metrics


def build_api_evidence_pack(config: Config, alert_record: dict[str, Any]) -> EvidencePack:
    """Build the HIGH-risk evidence payload used by the dashboard's API action."""
    window_end = alert_record["window_end"]
    if isinstance(window_end, str):
        window_end = date.fromisoformat(window_end)
    window_start = window_end - timedelta(days=config.default.time.window_days - 1)
    category = str(alert_record["category"])
    sku = str(alert_record["sku"])
    clean = read_category_table(
        Table.CLEAN_REVIEWS,
        config,
        category,
        start=window_start,
        end=window_end,
        skus=[sku],
    )
    try:
        nlp = read_category_table(
            Table.NLP_FEATURES,
            config,
            category,
            start=window_start,
            end=window_end,
            review_ids=clean["review_id"].to_list(),
        )
    except FileNotFoundError:
        nlp = pl.DataFrame()
    pack = build_evidence_pack(
        alert_record,
        clean,
        nlp_features_df=nlp,
        window_days=config.default.time.window_days,
        max_reviews=config.llm.max_top_reviews,
        mask_pii_enabled=getattr(config.llm, "mask_pii", True),
    )
    if not pack.reviews:
        raise ValueError(f"Không có review làm bằng chứng cho SKU {sku} tại {window_end}.")
    return pack


def indicator_breakdown(
    indicator_matrix: pl.DataFrame, config: Config, sku: str, as_of: date
) -> list[dict[str, Any]]:
    """Describe I1-I11 at a date with the same normalized scoring weights as RiskScore."""
    weights = calculate_weights(config)
    rows = (
        indicator_matrix.filter((pl.col("sku") == sku) & (pl.col("window_end") == as_of)).to_dicts()
        if not indicator_matrix.is_empty()
        else []
    )
    by_id = {row["indicator_id"]: row for row in rows}
    out = []
    for indicator_id, spec in config.indicators.indicators.items():
        row = by_id.get(indicator_id, {})
        triggered = bool(row.get("triggered", False))
        weight = weights.get(indicator_id, 0.0)
        out.append(
            {
                "indicator_id": indicator_id,
                "name": spec.name,
                "group": spec.group,
                "priority": spec.priority,
                "trigger": spec.trigger.model_dump(),
                "triggered": triggered,
                "strength": row.get("strength"),
                "weight": weight,
                "contribution": weight if triggered else 0.0,
            }
        )
    return out


def series_for(
    config: Config, category: str, sku: str, start: date, end: date
) -> dict[str, pl.DataFrame]:
    """Read feature windows and daily review counts for one selected SKU only."""
    features = read_category_table(
        Table.FEATURE_SERIES, config, category, start=start, end=end, skus=[sku]
    )
    daily = read_category_table(Table.DAILY_AGG, config, category, start=start, end=end, skus=[sku])
    indicators = read_category_table(
        Table.INDICATOR_MATRIX, config, category, start=start, end=end, skus=[sku]
    )
    alerts = load_alerts(config, category, start=start, end=end, skus=[sku])
    return {
        "features": features,
        "daily": daily,
        "indicators": indicators,
        "alerts": alerts,
    }


def evidence_reviews(
    config: Config,
    category: str,
    alert_record: dict[str, Any],
    *,
    indicator_id: str | None = None,
) -> list[dict[str, Any]]:
    """Return the exact representative reviews chosen by the LLM EvidencePack builder."""
    window_end = alert_record["window_end"]
    if isinstance(window_end, str):
        window_end = date.fromisoformat(window_end)
    window_start = window_end - timedelta(days=config.default.time.window_days - 1)
    clean = read_category_table(
        Table.CLEAN_REVIEWS,
        config,
        category,
        start=window_start,
        end=window_end,
        skus=[str(alert_record["sku"])],
    )
    if clean.is_empty():
        return []
    nlp = read_category_table(
        Table.NLP_FEATURES,
        config,
        category,
        start=window_start,
        end=window_end,
        review_ids=clean["review_id"].to_list(),
    )
    pack = build_evidence_pack(
        alert_record,
        clean,
        nlp_features_df=nlp,
        window_days=config.default.time.window_days,
        max_reviews=config.llm.max_top_reviews,
        mask_pii_enabled=getattr(config.llm, "mask_pii", True),
    )
    reviews = [review.model_dump() for review in pack.reviews]
    spam_by_id = dict(clean.select("review_id", "is_spam").iter_rows())
    nlp_by_id = {
        row["review_id"]: row
        for row in nlp.select(
            "review_id",
            *[column for column in ("sentiment_score", "mismatch") if column in nlp.columns],
        ).to_dicts()
    }
    for review in reviews:
        review["is_spam"] = bool(spam_by_id.get(review["review_id"], False))
        nlp_record = nlp_by_id.get(review["review_id"], {})
        review["sentiment_score"] = nlp_record.get("sentiment_score")
        review["mismatch"] = nlp_record.get("mismatch")
    if indicator_id == "I8":
        reviews = [review for review in reviews if review["risk_hit"]]
    elif indicator_id == "I3":
        reviews = [
            review
            for review in reviews
            if review["sentiment_score"] is not None and review["sentiment_score"] < 0
        ]
    elif indicator_id == "I4":
        reviews = [review for review in reviews if review["rating"] <= 2]
    elif indicator_id == "I6":
        reviews = [review for review in reviews if review["rating"] in (1, 5)]
    elif indicator_id == "I7":
        reviews = [review for review in reviews if review["risk_hit"]]
    elif indicator_id == "I9":
        reviews = [review for review in reviews if review["aspect"]]
    elif indicator_id == "I10":
        reviews = [review for review in reviews if review["mismatch"]]
    elif indicator_id == "I11":
        reviews = [review for review in reviews if review["verified_purchase"] is False]
    return reviews


def load_recommendation(
    config: Config, category: str, sku: str, window_end: date
) -> dict[str, Any]:
    """Load and validate a per-window or summary recommendation without raising on absence."""
    from brandsentinel.core.io import recommendation_path

    candidates = [
        recommendation_path(config, category, sku, window_end),
        config.path("data_processed")
        / "recommendations"
        / f"category={category}"
        / "recommendations.json",
    ]
    errors: list[str] = []
    for path in candidates:
        if not path.is_file():
            continue
        try:
            raw = read_json(path)
            values = raw if isinstance(raw, list) else [raw]
            for value in values:
                if value.get("sku") == sku and value.get("window_end") == window_end.isoformat():
                    recommendation = RecommendationOutput.model_validate(value)
                    return {
                        "status": "ok",
                        "recommendation": recommendation.model_dump(mode="json"),
                        "is_fallback": recommendation.is_fallback,
                        "path": path,
                    }
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            errors.append(f"{path.name}: {exc}")
    return {
        "status": "invalid" if errors else "missing",
        "recommendation": None,
        "is_fallback": False,
        "error": "; ".join(errors) if errors else None,
    }


def recommendation_summaries(config: Config, category: str) -> dict[tuple[str, str], str]:
    """Return available short summaries indexed by SKU and evidence window."""
    path = (
        config.path("data_processed")
        / "recommendations"
        / f"category={category}"
        / "recommendations.json"
    )
    if not path.is_file():
        return {}
    raw = read_json(path)
    values = raw if isinstance(raw, list) else [raw]
    result: dict[tuple[str, str], str] = {}
    for value in values:
        try:
            recommendation = RecommendationOutput.model_validate(value)
        except (ValueError, TypeError):
            continue
        result[(recommendation.sku, recommendation.window_end)] = recommendation.summary
    return result


def validation_summary(config: Config, category: str) -> dict[str, Any]:
    """Load recall labels and precomputed evaluation JSON; never calculate metrics here."""
    try:
        recall_labels = read_category_table(Table.RECALL_LABELS, config, category)
    except FileNotFoundError:
        recall_labels = empty_table(Table.RECALL_LABELS)
    evaluation_dir = config.path("data_processed") / "evaluation" / f"category={category}"
    reports: list[dict[str, Any]] = []
    errors: list[str] = []
    if evaluation_dir.is_dir():
        for path in sorted(evaluation_dir.glob("*.json")):
            try:
                report = read_json(path)
                if isinstance(report, dict):
                    reports.append({"file": path.name, "data": report})
                else:
                    errors.append(f"{path.name}: kết quả phải là JSON object")
            except (OSError, ValueError) as exc:
                errors.append(f"{path.name}: {exc}")
    return {"recall_labels": recall_labels, "reports": reports, "errors": errors}


def has_validation_reports(config: Config, category: str) -> bool:
    """Check for stored evaluation results without opening the optional validation tab."""
    directory = config.path("data_processed") / "evaluation" / f"category={category}"
    return directory.is_dir() and any(directory.glob("*.json"))
