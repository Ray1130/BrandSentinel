"""Helpers for finding corroborating active signal groups."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

from brandsentinel.core.config import Config


def active_groups(triggered_ids: Iterable[str], config: Config) -> list[str]:
    """Return counted groups meeting the configured active-indicator minimum."""
    indicators = config.indicators
    tiering = config.thresholds.tiering
    counts: Counter[str] = Counter()
    for indicator_id in set(triggered_ids):
        spec = indicators.indicators.get(indicator_id)
        if spec is not None and spec.enabled and spec.in_score:
            counts[spec.group] += 1
    return sorted(
        group
        for group in tiering.groups_counted
        if counts[group] >= tiering.min_active_indicators_per_group
    )
