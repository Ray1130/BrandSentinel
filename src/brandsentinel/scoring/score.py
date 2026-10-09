"""Convert an indicator matrix into validated, per-SKU risk alerts."""

from __future__ import annotations

from collections import defaultdict
from datetime import date
from typing import TypedDict

import polars as pl

from brandsentinel.core.config import Config
from brandsentinel.core.schemas import conform, empty_table, validate
from brandsentinel.core.types import Level, Table
from brandsentinel.scoring.groups import active_groups
from brandsentinel.scoring.rules import classify_level
from brandsentinel.scoring.state import apply_hysteresis
from brandsentinel.scoring.thresholds import (
    ScoreThresholds,
    thresholds_for_category,
)
from brandsentinel.scoring.weights import calculate_weights


class _ScoredWindow(TypedDict):
    category: str
    window_end: date
    score: float
    groups_active: list[str]
    triggered_ids: list[str]
    verify_flag: bool
    thresholds: ScoreThresholds


def score_indicators(
    indicator_matrix: pl.DataFrame,
    config: Config,
    *,
    reference_scores: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Score each SKU-window and return the `alerts` data-contract table.

    Optional `reference_scores` must contain `category` and `score`; without it, configured
    category overrides and fallback thresholds are used.
    """
    validate(Table.INDICATOR_MATRIX, indicator_matrix)
    if indicator_matrix.is_empty():
        return empty_table(Table.ALERTS)

    weights = calculate_weights(config)
    indicators = config.indicators
    flag_ids = {
        indicator_id
        for indicator_id in config.thresholds.flags.verify_authenticity_from
        if indicators.indicators[indicator_id].enabled
    }
    scored_ids = set(weights)
    thresholds_by_category = {
        category: thresholds_for_category(category, config, reference_scores)
        for category in indicator_matrix["category"].unique().to_list()
    }

    per_sku: dict[str, list[_ScoredWindow]] = defaultdict(list)
    for (sku, window_end), rows in indicator_matrix.partition_by(
        ["sku", "window_end"], as_dict=True
    ).items():
        category_values = rows["category"].unique().to_list()
        if len(category_values) != 1:
            raise ValueError(f"{sku}: category không nhất quán trong window {window_end}")
        category = category_values[0]
        triggered_ids = sorted(
            (row["indicator_id"] for row in rows.iter_rows(named=True) if row["triggered"]),
            key=lambda indicator_id: int(indicator_id[1:]),
        )
        triggered_set = set(triggered_ids)
        active_scored_ids = set(triggered_ids) & scored_ids
        score = sum(weights[indicator_id] for indicator_id in active_scored_ids)
        active = active_groups(active_scored_ids, config)
        verify_flag = (
            bool(triggered_set & flag_ids)
            if config.thresholds.flags.require == "any"
            else bool(flag_ids) and flag_ids.issubset(triggered_set)
        )
        per_sku[sku].append(
            {
                "category": category,
                "window_end": window_end,
                "score": score,
                "groups_active": active,
                "triggered_ids": triggered_ids,
                "verify_flag": verify_flag,
                "thresholds": thresholds_by_category[category],
            }
        )

    rows_out: list[dict[str, object]] = []
    for sku, windows in per_sku.items():
        windows.sort(key=lambda row: row["window_end"])
        candidates: list[Level] = []
        persistence: list[int] = []
        persist = 0
        for window in windows:
            thresholds = window["thresholds"]
            persist = min(persist + 1, 127) if float(window["score"]) >= thresholds.tau1 else 0
            persistence.append(persist)
            candidates.append(
                classify_level(
                    float(window["score"]),
                    thresholds,
                    persist,
                    window["groups_active"],
                    window["triggered_ids"],
                    config,
                )
            )

        levels = apply_hysteresis(candidates, config)
        for window, persist, level in zip(windows, persistence, levels, strict=True):
            rows_out.append(
                {
                    "sku": sku,
                    "category": window["category"],
                    "window_end": window["window_end"],
                    "score": window["score"],
                    "level": level.value,
                    "groups_active": window["groups_active"],
                    "persist": persist,
                    "verify_flag": window["verify_flag"],
                    "triggered_ids": window["triggered_ids"],
                }
            )

    result = pl.DataFrame(rows_out)
    return validate(Table.ALERTS, conform(Table.ALERTS, result)).sort(["sku", "window_end"])
