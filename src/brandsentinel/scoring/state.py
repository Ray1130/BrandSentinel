"""Per-SKU persistence counts and level hysteresis."""

from __future__ import annotations

from collections.abc import Iterable

from brandsentinel.core.config import Config
from brandsentinel.core.types import Level


def persistence_counts(scores: Iterable[float], tau1: float) -> list[int]:
    """Count consecutive windows at or above tau1, resetting on a lower score."""
    current = 0
    counts = []
    for score in scores:
        current = current + 1 if score >= tau1 else 0
        counts.append(min(current, 127))
    return counts


def apply_hysteresis(candidates: Iterable[Level], config: Config) -> list[Level]:
    """Delay level changes according to the configured per-SKU window state."""
    state = config.thresholds.state
    required = state.hysteresis.downgrade_after_windows
    current: Level | None = None
    pending: Level | None = None
    pending_count = 0
    result = []

    for candidate in candidates:
        if current is None:
            current = candidate
        elif candidate.rank > current.rank and state.escalate_immediately:
            current = candidate
            pending = None
            pending_count = 0
        elif candidate != current:
            if candidate == pending:
                pending_count += 1
            else:
                pending = candidate
                pending_count = 1
            if pending_count >= required:
                current = candidate
                pending = None
                pending_count = 0
        else:
            pending = None
            pending_count = 0
        result.append(current)
    return result
