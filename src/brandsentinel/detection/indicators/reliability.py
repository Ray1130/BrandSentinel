"""Review-authenticity indicators I10-I11."""

from __future__ import annotations

import numpy as np
import polars as pl

from ...core.registry import register
from ...core.types import Group
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


@register
class RatingTextMismatch(Indicator):
    """I10: mismatch rate tăng bất thường so với baseline, không tham gia RiskScore."""

    id, group = "I10", Group.FLAG

    def compute(self, feats: pl.DataFrame, baseline: Baseline) -> pl.DataFrame:
        del baseline
        return run_indicator(
            feats,
            "mismatch_rate",
            lambda _dates, values: values,
            _as_rule(self.trigger),
            z_kw={"scale_floor": float(self.params.get("scale_floor", 0.05))},
        )


@register
class VerifiedRatioDrop(Indicator):
    """I11: tỷ lệ verified giảm bất thường, đồng thời volume tăng nếu được cấu hình."""

    id, group = "I11", Group.FLAG

    def select_features(self, feature_series: pl.DataFrame) -> pl.DataFrame:
        required_ids = {"I11"}
        if self.params.get("require_volume_spike", True):
            required_ids.add("I1")
        return feature_series.filter(
            pl.col("indicator_id").is_in(sorted(required_ids))
            & pl.col("feature").is_in(["verified_ratio_all", "log1p_daily_count"])
        )

    def compute(self, feats: pl.DataFrame, baseline: Baseline) -> pl.DataFrame:
        del baseline
        verified = feats.filter(
            (pl.col("indicator_id") == "I11") & (pl.col("feature") == "verified_ratio_all")
        ).select(
            "sku",
            "category",
            "window_end",
            pl.col("raw_value").alias("verified_ratio"),
        )
        if verified.is_empty():
            return pl.DataFrame(schema=_RESULT_SCHEMA)

        require_volume = bool(self.params.get("require_volume_spike", True))
        if require_volume:
            volume = feats.filter(
                (pl.col("indicator_id") == "I1") & (pl.col("feature") == "log1p_daily_count")
            ).select(
                "sku",
                "category",
                "window_end",
                pl.col("raw_value").alias("volume_value"),
            )
            panel = verified.join(
                volume,
                on=["sku", "category", "window_end"],
                how="left",
            )
        else:
            panel = verified.with_columns(pl.lit(None, dtype=pl.Float64).alias("volume_value"))

        rule = _as_rule(self.trigger, direction="down")
        volume_k = float(self.params.get("volume_spike_k", 2.5))
        strength_by_key: dict[tuple[str, object], float] = {}
        triggered_by_key: dict[tuple[str, object], bool] = {}
        for part in panel.sort("window_end").partition_by("sku", maintain_order=True):
            verified_values = part["verified_ratio"].cast(pl.Float64).to_numpy()
            verified_z = rolling_robust_z(
                verified_values,
                scale_floor=float(self.params.get("verified_scale_floor", 0.02)),
            )
            if require_volume:
                volume_values = part["volume_value"].cast(pl.Float64).to_numpy()
                volume_z = rolling_robust_z(
                    volume_values,
                    scale_floor=float(self.params.get("volume_scale_floor", 0.05)),
                )
                verified_z[~(volume_z > volume_k)] = np.nan

            fired, strengths = m_of_n(verified_z, rule)
            keys = list(zip(part["sku"].to_list(), part["window_end"].to_list(), strict=True))
            for key, is_fired, strength in zip(keys, fired, strengths, strict=True):
                triggered_by_key[key] = bool(is_fired)
                strength_by_key[key] = float(strength)

        output = panel.select("sku", "category", "window_end")
        keys = list(zip(output["sku"].to_list(), output["window_end"].to_list(), strict=True))
        return output.with_columns(
            pl.Series("triggered", [triggered_by_key.get(key, False) for key in keys]),
            pl.Series("strength", [strength_by_key.get(key, 0.0) for key in keys]),
        ).select(list(_RESULT_SCHEMA))
