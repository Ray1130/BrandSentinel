"""Synthetic crisis injection and development-category calibration reports."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from itertools import product
from typing import Any

import polars as pl

from brandsentinel.core.config import Config
from brandsentinel.core.schemas import validate
from brandsentinel.core.types import Table
from brandsentinel.detection.stage import detect
from brandsentinel.scoring.score import score_indicators
from brandsentinel.scoring.thresholds import (
    thresholds_for_category,
)
from brandsentinel.scoring.weights import calculate_weights

_EVENT_INDICATORS = {
    "volume": ("I1", "I2"),
    "rating": ("I4", "I5", "I6"),
    "safety_keyword": ("I7", "I8", "I9"),
    "review_bombing": ("I11",),
}
_TUNED_INDICATORS = tuple(iid for ids in _EVENT_INDICATORS.values() for iid in ids)
_M_N_CANDIDATES = ((1, 3), (2, 3), (3, 5))
_K_MULTIPLIERS = (0.75, 1.0, 1.25)
_PELT_PENALTY_MULTIPLIERS = (0.75, 1.0, 1.25)


@dataclass(frozen=True)
class SyntheticCase:
    name: str
    sku: str
    event_start: date
    expected_indicators: tuple[str, ...]


@dataclass(frozen=True)
class SyntheticSuite:
    feature_series: pl.DataFrame
    control_sku: str
    cases: tuple[SyntheticCase, ...]


def inject_synthetic_crises(
    feature_series: pl.DataFrame,
    *,
    event_days: int = 14,
) -> SyntheticSuite:
    """Create four labeled crises and one no-injection control from category medians.

    This is a controlled detector check, not a substitute for labeled production incidents.
    """
    validate(Table.FEATURE_SERIES, feature_series)
    if feature_series.is_empty():
        raise ValueError("feature_series không được rỗng")
    if event_days < 1:
        raise ValueError("event_days phải >= 1")
    categories = feature_series["category"].unique().to_list()
    if len(categories) != 1:
        raise ValueError("chỉ hiệu chỉnh một category mỗi lần")

    template = (
        feature_series.group_by("window_end", "indicator_id", "feature", "category")
        .agg(
            pl.col("raw_value").median(),
            pl.col("n_window").median().round().cast(pl.Int32),
        )
        .sort(["window_end", "indicator_id", "feature"])
    )
    dates = template["window_end"].unique().sort().to_list()
    if len(dates) < max(60, event_days + 30):
        raise ValueError(
            f"cần tối thiểu {max(60, event_days + 30)} cửa sổ ngày để tách baseline và sự kiện"
        )
    event_start = dates[-event_days]
    source_features = set(template.select("indicator_id", "feature").rows())
    required = {
        ("I1", "log1p_daily_count"),
        ("I2", "growth_rate"),
        ("I4", "low_star_ratio_shrunk"),
        ("I5", "rating_rolling_variance"),
        ("I6", "share_1star"),
        ("I6", "share_5star"),
        ("I6", "extreme_share"),
        ("I7", "term_burst_score"),
        ("I8", "risk_hits"),
        ("I8", "risk_distinct_users"),
        ("I9", "aspect_neg_safety"),
        ("I11", "verified_ratio_all"),
    }
    missing = sorted(required - source_features)
    if missing:
        raise ValueError(f"feature_series thiếu đặc trưng cần cho synthetic injection: {missing}")

    baseline_values = {
        (row["indicator_id"], row["feature"]): row["raw_value"]
        for row in template.group_by("indicator_id", "feature")
        .agg(pl.col("raw_value").median())
        .to_dicts()
    }

    control_sku = "SYNTH_CONTROL"
    panel_parts = [template.with_columns(pl.lit(control_sku).alias("sku"))]
    cases: list[SyntheticCase] = []
    specs = (
        ("volume", (("I1", "log1p_daily_count"), ("I2", "growth_rate"))),
        (
            "rating",
            (
                ("I4", "low_star_ratio_shrunk"),
                ("I5", "rating_rolling_variance"),
                ("I6", "share_1star"),
                ("I6", "share_5star"),
                ("I6", "extreme_share"),
            ),
        ),
        (
            "safety_keyword",
            (
                ("I7", "term_burst_score"),
                ("I8", "risk_hits"),
                ("I8", "risk_distinct_users"),
                ("I9", "aspect_neg_safety"),
            ),
        ),
        (
            "review_bombing",
            (("I1", "log1p_daily_count"), ("I11", "verified_ratio_all")),
        ),
    )
    for name, modified_features in specs:
        sku = f"SYNTH_{name.upper()}"
        injected = template.with_columns(pl.lit(sku).alias("sku"))
        for indicator_id, feature in modified_features:
            baseline = baseline_values[(indicator_id, feature)]
            if name in {"volume", "review_bombing"} and feature == "log1p_daily_count":
                value = float(baseline or 0.0) + 1.5
            elif name == "volume" and feature == "growth_rate":
                value = max(float(baseline or 0.0), 0.0) + 2.0
            elif name == "rating" and feature == "low_star_ratio_shrunk":
                value = 0.9
            elif name == "rating" and feature == "rating_rolling_variance":
                value = 2.0
            elif name == "rating" and feature in {"share_1star", "share_5star"}:
                value = 0.4
            elif name == "rating" and feature == "extreme_share":
                value = 0.8
            elif name == "safety_keyword" and feature == "term_burst_score":
                value = max(float(baseline or 0.0) + 5.0, 5.0)
            elif name == "safety_keyword" and feature in {"risk_hits", "risk_distinct_users"}:
                value = 4.0
            elif name == "safety_keyword" and feature == "aspect_neg_safety":
                value = 0.95
            elif name == "review_bombing" and feature == "verified_ratio_all":
                value = 0.05
            else:
                raise ValueError(f"Không có synthetic injection rule cho {name}:{feature}")
            injected = injected.with_columns(
                pl.when(
                    (pl.col("indicator_id") == indicator_id)
                    & (pl.col("feature") == feature)
                    & (pl.col("window_end") >= event_start)
                )
                .then(pl.lit(value))
                .otherwise(pl.col("raw_value"))
                .alias("raw_value")
            )
        panel_parts.append(injected)
        cases.append(
            SyntheticCase(
                name=name,
                sku=sku,
                event_start=event_start,
                expected_indicators=_EVENT_INDICATORS[name],
            )
        )

    panel = pl.concat(panel_parts).sort(["sku", "window_end", "indicator_id", "feature"])
    validate(Table.FEATURE_SERIES, panel)
    return SyntheticSuite(panel, control_sku, tuple(cases))


def _score_series(matrix: pl.DataFrame, config: Config) -> pl.DataFrame:
    weights = calculate_weights(config)
    rows = []
    for (sku, window_end), part in matrix.partition_by(["sku", "window_end"], as_dict=True).items():
        category = part["category"][0]
        triggered = set(part.filter(pl.col("triggered"))["indicator_id"].to_list()) & set(weights)
        rows.append(
            {
                "sku": sku,
                "category": category,
                "window_end": window_end,
                "score": sum(weights[iid] for iid in triggered),
            }
        )
    return pl.DataFrame(
        rows,
        schema={
            "sku": pl.String,
            "category": pl.String,
            "window_end": pl.Date,
            "score": pl.Float64,
        },
    )


def _trigger_candidates(spec: Any) -> list[tuple[float, int, int]]:
    trigger = spec.trigger
    if trigger.k is None:
        return []
    m_n = {(trigger.m, trigger.n), *_M_N_CANDIDATES}
    return sorted(
        {
            (max(0.05, trigger.k * multiplier), m, n)
            for multiplier, (m, n) in product(_K_MULTIPLIERS, m_n)
            if m <= n
        }
    )


def _indicator_metrics(
    output: pl.DataFrame,
    indicator_id: str,
    suite: SyntheticSuite,
) -> tuple[float, float]:
    control = output.filter(pl.col("sku") == suite.control_sku)
    false_alarm_rate = float(control["triggered"].mean()) if control.height else 0.0
    recalls = []
    for case in suite.cases:
        if indicator_id not in case.expected_indicators:
            continue
        event = output.filter(
            (pl.col("sku") == case.sku) & (pl.col("window_end") >= case.event_start)
        )
        recalls.append(float(event["triggered"].mean()) if event.height else 0.0)
    return (sum(recalls) / len(recalls) if recalls else 0.0, false_alarm_rate)


def _candidate_config(
    config: Config,
    indicator_id: str,
    *,
    trigger_values: tuple[float, int, int] | None = None,
    safety_thresholds: tuple[int, int] | None = None,
    penalty_beta: float | None = None,
) -> Config:
    specs = dict(config.indicators.indicators)
    spec = specs[indicator_id]
    updates: dict[str, Any] = {}
    if trigger_values is not None:
        k, m, n = trigger_values
        updates["trigger"] = spec.trigger.model_copy(update={"k": k, "m": m, "n": n})
    if safety_thresholds is not None:
        hits, users = safety_thresholds
        updates["params"] = {
            **spec.params,
            "min_hits": hits,
            "min_distinct_users": users,
        }
    if updates:
        specs[indicator_id] = spec.model_copy(update=updates)
    indicators = config.indicators.model_copy(update={"indicators": specs})
    changes: dict[str, Any] = {"indicators": indicators}
    if penalty_beta is not None:
        change_point = config.default.changepoint.model_copy(update={"penalty_beta": penalty_beta})
        changes["default"] = config.default.model_copy(update={"changepoint": change_point})
    return config.model_copy(update=changes)


def _sweep_triggers(
    suite: SyntheticSuite,
    config: Config,
    *,
    max_false_alarm_rate: float,
) -> dict[str, dict[str, Any]]:
    # Importing stage registers the production detector implementations.
    from brandsentinel.core.registry import build_indicators

    results: dict[str, dict[str, Any]] = {}
    for indicator_id in _TUNED_INDICATORS:
        if indicator_id not in config.indicators.indicators:
            continue
        spec = config.indicators.indicators[indicator_id]
        candidates: list[Config] = []
        labels: list[dict[str, Any]] = []
        for k, m, n in _trigger_candidates(spec):
            betas = (
                [
                    config.default.changepoint.penalty_beta * multiplier
                    for multiplier in _PELT_PENALTY_MULTIPLIERS
                ]
                if indicator_id == "I5" and spec.params.get("use_pelt", True)
                else [None]
            )
            for beta in betas:
                candidates.append(
                    _candidate_config(
                        config,
                        indicator_id,
                        trigger_values=(k, m, n),
                        penalty_beta=beta,
                    )
                )
                labels.append({"k": k, "m": m, "n": n, "pelt_penalty_beta": beta})
        if indicator_id == "I8":
            for hits, users in product((1, 2, 3), repeat=2):
                candidates.append(
                    _candidate_config(
                        config,
                        indicator_id,
                        safety_thresholds=(hits, users),
                    )
                )
                labels.append({"min_hits": hits, "min_distinct_users": users})
        if not candidates:
            continue

        scored_candidates = []
        for label, candidate_config in zip(labels, candidates, strict=True):
            detector = build_indicators(candidate_config)[indicator_id]
            output = detector.run(suite.feature_series, baseline=None)
            recall, false_alarm_rate = _indicator_metrics(output, indicator_id, suite)
            scored_candidates.append(
                {
                    **label,
                    "recall": recall,
                    "false_alarm_rate": false_alarm_rate,
                    "meets_false_alarm_target": false_alarm_rate <= max_false_alarm_rate,
                }
            )
        eligible = [r for r in scored_candidates if r["meets_false_alarm_target"]]
        pool = eligible or scored_candidates
        pool.sort(
            key=lambda row: (
                row["recall"],
                -row["false_alarm_rate"],
                row.get("k") == spec.trigger.k
                and row.get("m") == spec.trigger.m
                and row.get("n") == spec.trigger.n,
            ),
            reverse=True,
        )
        results[indicator_id] = {
            "selection_note": (
                "best recall subject to false-alarm target"
                if eligible
                else "no candidate met false-alarm target; best recall shown"
            ),
            "selected": pool[0],
            "candidates": scored_candidates,
        }
    return results


def calibrate_category(
    feature_series: pl.DataFrame,
    config: Config,
    *,
    max_false_alarm_rate: float = 0.05,
    event_days: int = 14,
    tune_triggers: bool = True,
) -> dict[str, Any]:
    """Calibrate score thresholds and evaluate labeled synthetic crises for one category."""
    if not 0 <= max_false_alarm_rate <= 1:
        raise ValueError("max_false_alarm_rate phải nằm trong [0, 1]")
    validate(Table.FEATURE_SERIES, feature_series)
    if feature_series.is_empty():
        raise ValueError("feature_series không được rỗng")
    categories = feature_series["category"].unique().to_list()
    if len(categories) != 1:
        raise ValueError("chỉ hiệu chỉnh một category mỗi lần")
    category = categories[0]

    reference_matrix = detect(feature_series, config)
    reference_scores = _score_series(reference_matrix, config)
    threshold_reference_scores = reference_scores
    excluded_windows = 0
    if config.thresholds.calibration.reference_windows.exclude_changepoint_windows:
        changepoints = reference_matrix.filter(
            (pl.col("indicator_id") == "I5") & pl.col("triggered")
        ).select("sku", "window_end")
        threshold_reference_scores = (
            reference_scores.join(
                changepoints.with_columns(pl.lit(True).alias("_exclude")),
                on=["sku", "window_end"],
                how="left",
            )
            .filter(pl.col("_exclude").is_null())
            .drop("_exclude")
        )
        excluded_windows = reference_scores.height - threshold_reference_scores.height
    scores = threshold_reference_scores["score"].to_list()
    calibrated = thresholds_for_category(
        category,
        config,
        threshold_reference_scores.select("category", "score"),
    )
    suite = inject_synthetic_crises(feature_series, event_days=event_days)
    synthetic_matrix = detect(suite.feature_series, config)
    reference_by_category = threshold_reference_scores.select("category", "score")
    alerts = score_indicators(
        synthetic_matrix,
        config,
        reference_scores=reference_by_category,
    )

    case_reports = []
    for case in suite.cases:
        signal_rows = synthetic_matrix.filter(pl.col("sku") == case.sku)
        event_rows = signal_rows.filter(pl.col("window_end") >= case.event_start)
        before_rows = signal_rows.filter(pl.col("window_end") < case.event_start)
        case_alerts = alerts.filter(
            (pl.col("sku") == case.sku) & (pl.col("window_end") >= case.event_start)
        )
        control_indicator_false_alarm_rates = {}
        for indicator_id in case.expected_indicators:
            control_indicator_rows = synthetic_matrix.filter(
                (pl.col("sku") == suite.control_sku) & (pl.col("indicator_id") == indicator_id)
            )
            control_indicator_false_alarm_rates[indicator_id] = (
                float(control_indicator_rows["triggered"].mean())
                if control_indicator_rows.height
                else 0.0
            )
        expected = {}
        for indicator_id in case.expected_indicators:
            rows = event_rows.filter(pl.col("indicator_id") == indicator_id)
            expected[indicator_id] = float(rows["triggered"].mean()) if rows.height else 0.0
        bombing_rows = event_rows.filter(pl.col("indicator_id") == "I11")
        case_reports.append(
            {
                "name": case.name,
                "event_start": case.event_start.isoformat(),
                "windows": event_rows["window_end"].n_unique(),
                "indicator_recall": expected,
                "control_indicator_false_alarm_rate": control_indicator_false_alarm_rates,
                "pre_event_indicator_false_alarm_rate": (
                    float(before_rows["triggered"].mean()) if before_rows.height else 0.0
                ),
                "alert_windows": case_alerts.height,
                "captured_by_alerts": case_alerts.height > 0,
                "review_bombing_flagged_windows": (
                    int(bombing_rows["triggered"].sum()) if bombing_rows.height else 0
                ),
                "alert_verify_flagged_windows": (
                    int(case_alerts["verify_flag"].sum()) if case_alerts.height else 0
                ),
            }
        )

    control_alerts = alerts.filter(pl.col("sku") == suite.control_sku)
    control_windows = suite.feature_series.filter(pl.col("sku") == suite.control_sku)[
        "window_end"
    ].n_unique()
    empirical_false_alarm_rate = control_alerts.height / control_windows if control_windows else 0.0
    captured_cases = sum(case_report["captured_by_alerts"] for case_report in case_reports)
    report: dict[str, Any] = {
        "category": category,
        "synthetic": True,
        "synthetic_note": (
            "Synthetic cases are detector checks, not independently labeled incidents. "
            "The control false-alarm rate is only a proxy; production precision/recall "
            "requires manually verified incident labels."
        ),
        "source_windows": reference_scores.height,
        "threshold_reference_windows": len(scores),
        "excluded_changepoint_windows": excluded_windows,
        "thresholds": {
            "method": (
                "category_override"
                if category in config.thresholds.overrides
                else config.thresholds.calibration.method
            ),
            "tau1_percentile": config.thresholds.calibration.tau1_percentile,
            "tau2_percentile": config.thresholds.calibration.tau2_percentile,
            "tau1": calibrated.tau1,
            "tau2": calibrated.tau2,
            "used_fallback": (
                category not in config.thresholds.overrides
                and len(scores) < config.thresholds.calibration.reference_windows.min_windows
            ),
            "floors": {
                "tau1": config.thresholds.calibration.floors.tau1_min,
                "tau2": config.thresholds.calibration.floors.tau2_min,
            },
        },
        "synthetic_cases": case_reports,
        "event_capture": {
            "captured_cases": captured_cases,
            "total_cases": len(case_reports),
            "rate": captured_cases / len(case_reports) if case_reports else 0.0,
        },
        "control": {
            "windows": control_windows,
            "alert_windows": control_alerts.height,
            "alert_false_alarm_rate": empirical_false_alarm_rate,
            "target_false_alarm_rate": max_false_alarm_rate,
            "meets_target": empirical_false_alarm_rate <= max_false_alarm_rate,
        },
        "trigger_calibration": (
            _sweep_triggers(
                suite,
                config,
                max_false_alarm_rate=max_false_alarm_rate,
            )
            if tune_triggers
            else {}
        ),
        "writes_config": False,
    }
    return report
