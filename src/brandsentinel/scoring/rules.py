"""Risk tiering rules based on score, persistence, group confirmation, and severe evidence."""

from __future__ import annotations

from collections.abc import Iterable

from brandsentinel.core.config import Config
from brandsentinel.core.types import Level
from brandsentinel.scoring.thresholds import ScoreThresholds


def classify_level(
    score: float,
    thresholds: ScoreThresholds,
    persist: int,
    groups_active: Iterable[str],
    triggered_ids: Iterable[str],
    config: Config,
) -> Level:
    """Apply configured MEDIUM/HIGH confirmation rules to a single scored window."""
    tiering = config.thresholds.tiering
    groups = set(groups_active)
    ids = set(triggered_ids)
    severe_override = (
        tiering.high.allow_severe_override and "I8" in ids and "I8" in config.indicators.scored_ids
    )
    high = score >= thresholds.tau2 and (
        len(groups) >= tiering.high.min_groups_active or severe_override
    )
    medium = score >= thresholds.tau1 and (
        persist >= tiering.medium.min_persist_windows
        or len(groups) >= tiering.medium.min_groups_active
    )
    if high:
        return Level.HIGH
    if medium:
        return Level.MEDIUM
    return Level.LOW
