"""Content risk indicators I7-I9."""

from __future__ import annotations

import numpy as np
import polars as pl

from ...core.registry import register
from ...core.types import ASPECTS, Group
from ..base import Baseline, Indicator
from ..baseline import rolling_robust_z
from ..triggers import m_of_n
from .volume import _as_rule, run_indicator

_RESULT_SCHEMA = {
    "sku": pl.String,
    "category": pl.String,
    "window_end": pl.Date,
    "triggered": pl.Boolean,
    "strength": pl.Float64,
}


def _identity_series(dates: pl.Series, values: np.ndarray) -> np.ndarray:
    del dates
    return values


@register
class TopicBurst(Indicator):
    """I7: score bùng nổ chủ đề vượt baseline trong luật m-of-n."""

    id, group = "I7", Group.CONTENT

    def compute(self, feats: pl.DataFrame, baseline: Baseline) -> pl.DataFrame:
        del baseline
        return run_indicator(
            feats,
            "term_burst_score",
            _identity_series,
            _as_rule(self.trigger),
            z_kw={"scale_floor": float(self.params.get("scale_floor", 0.05))},
        )


@register
class SafetyRisk(Indicator):
    """I8: kích hoạt theo cả số review rủi ro và số người dùng độc lập.

    `strength` là 0.5 cho kích hoạt thường và 1.0 khi đạt cả hai ngưỡng nghiêm trọng,
    cho phép stage scoring áp dụng severe override mà không mở rộng hợp đồng indicator_matrix.
    """

    id, group = "I8", Group.CONTENT
    NORMAL_STRENGTH = 0.5
    SEVERE_STRENGTH = 1.0

    def compute(self, feats: pl.DataFrame, baseline: Baseline) -> pl.DataFrame:
        del baseline
        if feats.is_empty():
            return pl.DataFrame(schema=_RESULT_SCHEMA)

        wide = feats.select("sku", "category", "window_end", "feature", "raw_value").pivot(
            on="feature",
            index=["sku", "category", "window_end"],
            values="raw_value",
        )
        required = {"risk_hits", "risk_distinct_users"}
        if not required.issubset(wide.columns):
            return pl.DataFrame(schema=_RESULT_SCHEMA)

        min_hits = float(self.params.get("min_hits", 2))
        min_users = float(self.params.get("min_distinct_users", 2))
        severe_hits = float(self.params.get("severe_min_reviews", 3))
        severe_users = float(self.params.get("severe_min_distinct_users", 3))
        hits = wide["risk_hits"].cast(pl.Float64).to_numpy()
        users = wide["risk_distinct_users"].cast(pl.Float64).to_numpy()
        valid = np.isfinite(hits) & np.isfinite(users)
        triggered = valid & (hits >= min_hits) & (users >= min_users)
        severe = triggered & (hits >= severe_hits) & (users >= severe_users)
        strength = np.where(
            triggered,
            np.where(severe, self.SEVERE_STRENGTH, self.NORMAL_STRENGTH),
            0.0,
        )
        return (
            wide.select("sku", "category", "window_end")
            .with_columns(
                pl.Series("triggered", triggered),
                pl.Series("strength", strength),
            )
            .select(list(_RESULT_SCHEMA))
        )


@register
class AspectNegativity(Indicator):
    """I9: bất thường tiêu cực theo từng aspect; một aspect không bù trừ aspect khác."""

    id, group = "I9", Group.CONTENT

    def compute(self, feats: pl.DataFrame, baseline: Baseline) -> pl.DataFrame:
        del baseline
        if feats.is_empty():
            return pl.DataFrame(schema=_RESULT_SCHEMA)

        aspects = self.params.get("aspects", list(ASPECTS))
        unknown = set(aspects) - set(ASPECTS)
        if unknown:
            raise ValueError(f"I9 có aspect không hỗ trợ: {sorted(unknown)}")
        features = [f"aspect_neg_{aspect}" for aspect in aspects]
        selected = feats.filter(pl.col("feature").is_in(features)).select(
            "sku", "category", "window_end", "n_window", "feature", "raw_value"
        )
        keys = ["sku", "category", "window_end"]
        wide = selected.select(*keys, "feature", "raw_value").pivot(
            on="feature", index=keys, values="raw_value"
        )
        supports = selected.select(
            *keys,
            pl.col("feature").str.replace("aspect_neg_", "n_").alias("feature"),
            pl.col("n_window").cast(pl.Float64).alias("raw_value"),
        ).pivot(on="feature", index=keys, values="raw_value")
        wide = wide.join(supports, on=keys, how="left").sort(["sku", "window_end"])
        if wide.is_empty():
            return pl.DataFrame(schema=_RESULT_SCHEMA)

        rule = _as_rule(self.trigger)
        min_reviews = int(self.params.get("min_aspect_reviews", 5))
        scale_floor = float(self.params.get("scale_floor", 0.05))
        fired_by_window: dict[tuple[str, object], bool] = {}
        strength_by_window: dict[tuple[str, object], float] = {}

        for part in wide.partition_by("sku", maintain_order=True):
            keys = list(zip(part["sku"].to_list(), part["window_end"].to_list(), strict=True))
            for feature in features:
                aspect = feature.removeprefix("aspect_neg_")
                support = f"n_{aspect}"
                if feature not in part.columns or support not in part.columns:
                    continue
                values = part[feature].cast(pl.Float64).to_numpy()
                z = rolling_robust_z(values, scale_floor=scale_floor)
                enough_reviews = part[support].cast(pl.Float64).to_numpy() >= min_reviews
                z[~enough_reviews] = np.nan
                fired, strength = m_of_n(z, rule)
                for key, is_fired, value in zip(keys, fired, strength, strict=True):
                    fired_by_window[key] = fired_by_window.get(key, False) or bool(is_fired)
                    strength_by_window[key] = max(strength_by_window.get(key, 0.0), float(value))

        keys = list(zip(wide["sku"].to_list(), wide["window_end"].to_list(), strict=True))
        results = wide.select("sku", "category", "window_end").with_columns(
            pl.Series("triggered", [fired_by_window.get(key, False) for key in keys]),
            pl.Series("strength", [strength_by_window.get(key, 0.0) for key in keys]),
        )
        return results.select(list(_RESULT_SCHEMA))
