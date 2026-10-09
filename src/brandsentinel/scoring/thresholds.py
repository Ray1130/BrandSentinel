"""RiskScore threshold calibration and category-specific threshold lookup."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import polars as pl

from brandsentinel.core.config import Config


@dataclass(frozen=True)
class ScoreThresholds:
    tau1: float
    tau2: float


def calibrate_thresholds(scores: Sequence[float], config: Config) -> ScoreThresholds:
    """Calibrate thresholds from reference scores, or use configured fallback when too few."""
    calibration = config.thresholds.calibration
    fallback = calibration.fallback
    values = [float(score) for score in scores]
    if any(not math.isfinite(score) or not 0 <= score <= 1 for score in values):
        raise ValueError("reference scores phải hữu hạn và nằm trong [0, 1]")
    if len(values) < calibration.reference_windows.min_windows:
        return ScoreThresholds(tau1=fallback.tau1, tau2=fallback.tau2)

    floors = calibration.floors
    tau1 = max(
        float(np.percentile(values, calibration.tau1_percentile)),
        floors.tau1_min,
    )
    tau2 = max(
        float(np.percentile(values, calibration.tau2_percentile)),
        floors.tau2_min,
    )
    # Discrete score distributions can yield identical percentiles; preserve a usable HIGH band.
    tau1 = min(tau1, 1.0 - 1e-9)
    tau2 = min(max(tau2, tau1 + 1e-9), 1.0)
    return ScoreThresholds(tau1=tau1, tau2=tau2)


def thresholds_for_category(
    category: str,
    config: Config,
    reference_scores: pl.DataFrame | None = None,
) -> ScoreThresholds:
    """Resolve overrides or calibrate from a `category, score` reference table."""
    thresholds = config.thresholds
    if category in thresholds.overrides:
        override = thresholds.overrides[category]
        return ScoreThresholds(tau1=override.tau1, tau2=override.tau2)
    if reference_scores is None:
        return calibrate_thresholds([], config)
    if not {"category", "score"}.issubset(reference_scores.columns):
        raise ValueError("reference_scores cần có cột `category` và `score`")
    scores = reference_scores
    if thresholds.calibration.per_category:
        scores = scores.filter(pl.col("category") == category)
    return calibrate_thresholds(scores["score"].drop_nulls().to_list(), config)
