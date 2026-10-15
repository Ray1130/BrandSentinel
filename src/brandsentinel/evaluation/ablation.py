"""Signal group ablation and threshold sensitivity analysis on verified recall events."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import polars as pl

from brandsentinel.core.config import Config, get_config
from brandsentinel.core.schemas import conform, empty_table, validate
from brandsentinel.core.types import Level, Table
from brandsentinel.evaluation.metrics import evaluate_alerts
from brandsentinel.scoring.rules import classify_level
from brandsentinel.scoring.thresholds import ScoreThresholds, thresholds_for_category
from brandsentinel.scoring.weights import calculate_weights

DEFAULT_TAU_PAIRS: list[tuple[float, float]] = [
    (0.15, 0.30),
    (0.20, 0.35),
    (0.20, 0.40),
    (0.25, 0.45),
    (0.30, 0.50),
]


def ablate_alerts(
    alerts: pl.DataFrame,
    *,
    drop_group: str | None = None,
    keep_only_group: str | None = None,
    config: Config | None = None,
) -> pl.DataFrame:
    """Return re-scored and re-tiered alerts when dropping or keeping a specific signal group.

    Parameters
    ----------
    alerts:
        Input alerts table conforming to Table.ALERTS data contract.
    drop_group:
        Name of signal group to remove (e.g. 'volume', 'rating', 'content', 'flag').
    keep_only_group:
        If specified, drops all signal groups except this one.
    config:
        BrandSentinel Config instance; uses default if omitted.
    """
    validate(Table.ALERTS, alerts)
    if alerts.is_empty():
        return empty_table(Table.ALERTS)

    if drop_group is not None and keep_only_group is not None:
        raise ValueError("Chỉ được chỉ định một trong hai: drop_group hoặc keep_only_group")

    cfg = config or get_config()
    weights = calculate_weights(cfg)
    group_by_indicator = {
        indicator_id: meta.group for indicator_id, meta in cfg.indicators.indicators.items()
    }

    thresholds_by_cat = {
        cat: thresholds_for_category(cat, cfg) for cat in alerts["category"].unique().to_list()
    }

    # Gom nhóm các window theo SKU để tính toán persistence liên tục theo thời gian
    per_sku: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in alerts.iter_rows(named=True):
        per_sku[row["sku"]].append(row)

    rows_out: list[dict[str, Any]] = []

    for sku, windows in per_sku.items():
        windows.sort(key=lambda w: w["window_end"])
        persist = 0

        for window in windows:
            cat = window["category"]
            thresholds = thresholds_by_cat[cat]
            orig_triggered: list[str] = window["triggered_ids"] or []
            orig_groups: list[str] = window["groups_active"] or []

            # 1. Lọc chỉ báo theo nhóm
            if keep_only_group is not None:
                new_triggered = [
                    iid for iid in orig_triggered if group_by_indicator.get(iid) == keep_only_group
                ]
                new_groups = [g for g in orig_groups if g == keep_only_group]
            elif drop_group is not None:
                new_triggered = [
                    iid for iid in orig_triggered if group_by_indicator.get(iid) != drop_group
                ]
                new_groups = [g for g in orig_groups if g != drop_group]
            else:
                new_triggered = list(orig_triggered)
                new_groups = list(orig_groups)

            # 2. Tính lại điểm RiskScore dựa trên các chỉ báo còn lại
            new_score = sum(weights[iid] for iid in new_triggered if iid in weights)

            # 3. Kiểm tra bằng chứng rủi ro nghiêm trọng (I8)
            # Nếu bỏ nhóm content, I8 bị loại -> không còn severe_evidence
            # Nếu giữ content và I8 kích hoạt, giữ trạng thái nếu trước đó là HIGH với < 3 nhóm
            severe_evidence = (
                "I8" in new_triggered
                and drop_group != "content"
                and (window["level"] == "HIGH" and len(orig_groups) < 3)
            )

            # 4. Cập nhật persistence đếm chuỗi cửa sổ >= tau1
            persist = min(persist + 1, 127) if new_score >= thresholds.tau1 else 0

            # 5. Phân cấp lại mức độ rủi ro (LOW / MEDIUM / HIGH)
            new_level = classify_level(
                new_score,
                thresholds,
                persist,
                new_groups,
                new_triggered,
                cfg,
                severe_evidence=severe_evidence,
            )

            if new_level == Level.LOW:
                continue

            verify_flag = window["verify_flag"]
            if drop_group in ("flag", None) and keep_only_group not in ("flag", None):
                verify_flag = False

            rows_out.append(
                {
                    "sku": sku,
                    "category": cat,
                    "window_end": window["window_end"],
                    "score": round(new_score, 6),
                    "level": new_level.value,
                    "groups_active": new_groups,
                    "persist": persist,
                    "verify_flag": verify_flag,
                    "triggered_ids": new_triggered,
                }
            )

    if not rows_out:
        return empty_table(Table.ALERTS)

    result = pl.DataFrame(rows_out)
    return validate(Table.ALERTS, conform(Table.ALERTS, result)).sort(["sku", "window_end"])


def reclassify_with_tau(
    alerts: pl.DataFrame,
    tau1: float,
    tau2: float,
    *,
    config: Config | None = None,
) -> pl.DataFrame:
    """Reclassify alert levels using arbitrary (tau1, tau2) threshold boundaries."""
    validate(Table.ALERTS, alerts)
    if alerts.is_empty():
        return empty_table(Table.ALERTS)

    if not (0.0 <= tau1 < tau2 <= 1.0):
        raise ValueError(
            f"Ngưỡng không hợp lệ: cần 0 <= tau1 < tau2 <= 1, nhận tau1={tau1}, tau2={tau2}"
        )

    cfg = config or get_config()
    thresholds = ScoreThresholds(tau1=tau1, tau2=tau2)

    per_sku: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in alerts.iter_rows(named=True):
        per_sku[row["sku"]].append(row)

    rows_out: list[dict[str, Any]] = []

    for sku, windows in per_sku.items():
        windows.sort(key=lambda w: w["window_end"])
        persist = 0

        for window in windows:
            score = float(window["score"])
            orig_groups: list[str] = window["groups_active"] or []
            orig_triggered: list[str] = window["triggered_ids"] or []

            severe_evidence = "I8" in orig_triggered and (
                window["level"] == "HIGH" and len(orig_groups) < 3
            )

            persist = min(persist + 1, 127) if score >= thresholds.tau1 else 0

            new_level = classify_level(
                score,
                thresholds,
                persist,
                orig_groups,
                orig_triggered,
                cfg,
                severe_evidence=severe_evidence,
            )

            if new_level == Level.LOW:
                continue

            rows_out.append(
                {
                    "sku": sku,
                    "category": window["category"],
                    "window_end": window["window_end"],
                    "score": score,
                    "level": new_level.value,
                    "groups_active": orig_groups,
                    "persist": persist,
                    "verify_flag": window["verify_flag"],
                    "triggered_ids": orig_triggered,
                }
            )

    if not rows_out:
        return empty_table(Table.ALERTS)

    result = pl.DataFrame(rows_out)
    return validate(Table.ALERTS, conform(Table.ALERTS, result)).sort(["sku", "window_end"])


def evaluate_signal_ablation(
    alerts: pl.DataFrame,
    recall_labels: pl.DataFrame,
    *,
    config: Config | None = None,
    lookback_days: int = 90,
    include_single_group: bool = False,
) -> pl.DataFrame:
    """Run an ablation study evaluating detection loss when removing each signal group."""
    validate(Table.ALERTS, alerts)
    validate(Table.RECALL_LABELS, recall_labels)

    cfg = config or get_config()

    configs_to_test: list[tuple[str, pl.DataFrame]] = [
        ("Full Model", alerts),
        ("w/o volume", ablate_alerts(alerts, drop_group="volume", config=cfg)),
        ("w/o rating", ablate_alerts(alerts, drop_group="rating", config=cfg)),
        ("w/o content", ablate_alerts(alerts, drop_group="content", config=cfg)),
    ]

    if include_single_group:
        configs_to_test.extend(
            [
                ("only volume", ablate_alerts(alerts, keep_only_group="volume", config=cfg)),
                ("only rating", ablate_alerts(alerts, keep_only_group="rating", config=cfg)),
                ("only content", ablate_alerts(alerts, keep_only_group="content", config=cfg)),
            ]
        )

    # 1. Tính toán kết quả cho từng cấu hình
    records: list[dict[str, Any]] = []
    base_recall: float | None = None
    base_lead_time: float | None = None

    for name, current_alerts in configs_to_test:
        metrics = evaluate_alerts(current_alerts, recall_labels, lookback_days=lookback_days)

        rec = metrics.get("event_recall")
        mean_lt = metrics.get("mean_lead_time_days")

        if name == "Full Model":
            base_recall = rec
            base_lead_time = mean_lt

        delta_recall = (
            round((rec - base_recall) * 100, 2)
            if (rec is not None and base_recall is not None)
            else None
        )
        delta_lead_time = (
            round(mean_lt - base_lead_time, 2)
            if (mean_lt is not None and base_lead_time is not None)
            else None
        )

        records.append(
            {
                "configuration": name,
                "recall_events": metrics["recall_events"],
                "detected_events": metrics["detected_events"],
                "missed_events": metrics["missed_events"],
                "event_recall": round(rec * 100, 2) if rec is not None else None,
                "mean_lead_time_days": round(mean_lt, 1) if mean_lt is not None else None,
                "median_lead_time_days": metrics["median_lead_time_days"],
                "alerts_in_window": metrics["alerts_in_window"],
                "total_alerts": current_alerts.height,
                "delta_recall_pct": delta_recall,
                "delta_lead_time_days": delta_lead_time,
            }
        )

    return pl.DataFrame(records)


def evaluate_tau_sensitivity(
    alerts: pl.DataFrame,
    recall_labels: pl.DataFrame,
    *,
    config: Config | None = None,
    tau_pairs: list[tuple[float, float]] | None = None,
    lookback_days: int = 90,
    high_only: bool = False,
) -> pl.DataFrame:
    """Evaluate event recall and alert volume sensitivity across different (tau1, tau2) pairs."""
    validate(Table.ALERTS, alerts)
    validate(Table.RECALL_LABELS, recall_labels)

    cfg = config or get_config()
    pairs = tau_pairs or DEFAULT_TAU_PAIRS

    records: list[dict[str, Any]] = []

    for tau1, tau2 in pairs:
        reclassified = reclassify_with_tau(alerts, tau1, tau2, config=cfg)
        eval_alerts = reclassified.filter(pl.col("level") == "HIGH") if high_only else reclassified

        metrics = evaluate_alerts(eval_alerts, recall_labels, lookback_days=lookback_days)
        rec = metrics.get("event_recall")
        mean_lt = metrics.get("mean_lead_time_days")

        label = (
            f"tau1={tau1:.2f}, tau2={tau2:.2f} (HIGH only)"
            if high_only
            else f"tau1={tau1:.2f}, tau2={tau2:.2f}"
        )

        records.append(
            {
                "setting": label,
                "tau1": tau1,
                "tau2": tau2,
                "recall_events": metrics["recall_events"],
                "detected_events": metrics["detected_events"],
                "missed_events": metrics["missed_events"],
                "event_recall": round(rec * 100, 2) if rec is not None else None,
                "mean_lead_time_days": round(mean_lt, 1) if mean_lt is not None else None,
                "median_lead_time_days": metrics["median_lead_time_days"],
                "alerts_in_window": metrics["alerts_in_window"],
                "total_alerts": eval_alerts.height,
            }
        )

    return pl.DataFrame(records)


def format_report_table(
    df: pl.DataFrame,
    *,
    title: str = "",
    format_type: str = "markdown",
) -> str:
    """Render a clean GitHub Markdown or LaTeX table for project evaluation reports."""
    if format_type not in ("markdown", "latex"):
        raise ValueError(
            f"Định dạng không hỗ trợ: {format_type}. Chỉ hỗ trợ 'markdown' hoặc 'latex'."
        )

    if df.is_empty():
        return f"### {title}\n(Không có dữ liệu)" if title else "(Không có dữ liệu)"

    headers = df.columns
    rows = df.iter_rows()

    if format_type == "markdown":
        lines: list[str] = []
        if title:
            lines.append(f"### {title}\n")

        # Tiêu đề bảng
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

        for row in rows:
            formatted_cells = [
                "N/A" if val is None else f"{val:.1f}" if isinstance(val, float) else str(val)
                for val in row
            ]
            lines.append("| " + " | ".join(formatted_cells) + " |")

        return "\n".join(lines)

    # Định dạng LaTeX
    lines = [f"% Table: {title}"] if title else []
    col_spec = "l" + "r" * (len(headers) - 1)
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\hline")
    escaped_headers = [h.replace("_", "\\_") for h in headers]
    lines.append(" & ".join(escaped_headers) + " \\\\")
    lines.append("\\hline")

    for row in rows:
        formatted_cells = [
            "-" if val is None else f"{val:.1f}" if isinstance(val, float) else str(val)
            for val in row
        ]
        lines.append(" & ".join(formatted_cells) + " \\\\")

    lines.append("\\hline")
    lines.append("\\end{tabular}")
    return "\n".join(lines)
