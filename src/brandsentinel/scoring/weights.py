"""Derive normalized scoring weights from indicator priorities."""

from __future__ import annotations

from brandsentinel.core.config import Config


def calculate_weights(config: Config) -> dict[str, float]:
    """Return normalized weights for enabled indicators that contribute to RiskScore."""
    indicators = config.indicators
    points = {
        indicator_id: indicators.priority_points[indicators.indicators[indicator_id].priority]
        for indicator_id in indicators.scored_ids
    }
    total = sum(points.values())
    if total <= 0:
        raise ValueError("RiskScore cần tổng priority_points lớn hơn 0")
    return {indicator_id: value / total for indicator_id, value in points.items()}
