"""Nhom rating: I4 tang ty le 1-2 sao, I5 bien dong rating, I6 phan cuc 1 sao va 5 sao.

Dung chung `run_indicator`, `_as_rule` voi volume.py. GIA DINH ve base.py nhu volume.py
(Indicator co id, group, trigger, params; compute(feats, baseline) tra bang ket qua chi bao).
"""

from __future__ import annotations

import numpy as np
import polars as pl

from ...core.registry import register
from ...core.types import Group
from ..base import Indicator
from ..baseline import fit_baseline, rolling_robust_z
from ..changepoint import ChangepointConfig, detect_changepoints
from ..triggers import TriggerRule, m_of_n
from .volume import _as_rule, run_indicator

F_LOW, F_VAR = "low_star_ratio_shrunk", "rating_rolling_variance"
F_S1, F_S5, F_EXT = "share_1star", "share_5star", "extreme_share"

_SCHEMA = {
    "sku": pl.Utf8,
    "category": pl.Utf8,
    "window_end": pl.Date,
    "triggered": pl.Boolean,
    "strength": pl.Float64,
}


@register
class LowStarIncrease(Indicator):
    """I4: z cua ty le 1-2 sao (da co ve nganh) + MUC TANG TOI THIEU.

    z > k  <=>  x - median > k * scale. Dat scale >= min_increase / k thi x - median > min_increase,
    nen khong can buoc rieng: SKU rat on dinh khong bao dong vi thay doi vai phan tram.
    """

    id, group = "I4", Group.RATING

    def compute(self, feats, baseline):
        rule = _as_rule(self.trigger)
        floor = max(
            float(self.params.get("scale_floor", 0.0)),
            float(self.params.get("min_increase", 0.05)) / rule.k,
        )
        return run_indicator(feats, F_LOW, lambda d, v: v, rule, z_kw={"scale_floor": floor})


@register
class RatingVolatility(Indicator):
    """I5: z cua phuong sai rating truot (chi chieu TANG) + xac nhan PELT neu use_pelt.

    PELT chi chay o cac ngay da qua luat z, tren chuoi phuong sai cat den ngay do.
    Sigma lay tu baseline (khong uoc luong tu sai phan cua chuoi truot chong lan).
    Z bao cao bat thuong; PELT xac nhan do la dich muc co thoi diem bat dau.
    """

    id, group = "I5", Group.RATING

    def compute(self, feats, baseline):
        rule = _as_rule(self.trigger, direction="up")
        floor = float(self.params.get("scale_floor", 0.05))
        lookback, gap = int(self.params.get("lookback", 90)), int(self.params.get("gap", 7))
        confirm = None
        if self.params.get("use_pelt", True):
            cp = self.config.default.changepoint
            cp_cfg = ChangepointConfig(
                model=cp.cost_model,
                min_size=cp.min_size,
                jump=cp.jump,
                pen_scale=cp.penalty_beta,
                max_window_days=cp.max_window_days,
                accept_recent_days=cp.accept_recent_days,
                min_effect_sigma=cp.min_effect_sigma,
                direction="up",
            )

            def confirm(series, fired):
                out = fired.copy()
                for t in np.flatnonzero(fired):
                    hi = t - gap
                    b = fit_baseline(
                        series[max(0, hi - lookback + 1) : hi + 1], min_n=14, scale_floor=floor
                    )
                    out[t] = bool(
                        detect_changepoints(
                            series[: t + 1], cfg=cp_cfg, sigma=b.scale if b else None
                        )
                    )
                return out

        return run_indicator(
            feats,
            F_VAR,
            lambda d, v: v,
            rule,
            confirm=confirm,
            z_kw={"scale_floor": floor, "lookback": lookback, "gap": gap},
        )


@register
class RatingPolarization(Indicator):
    """I6: ca hai cuc cung >= p_min (mac dinh 15%) va du mau (n_window >= min_n), giu m-of-n.

    Rui ro: SKU binh thuong co the co phan phoi chu U vuot 15%/15%.
    `min_increase_z` (None = tat) yeu cau extreme_share cao hon baseline cua SKU.
    """

    id, group = "I6", Group.RATING

    def compute(self, feats, baseline):
        p_min = float(self.params.get("p_min", 0.15))
        min_n = int(self.params.get("min_n", 20))
        zk = self.params.get("min_increase_z")
        rule = _as_rule(self.trigger)
        wide = feats.filter(pl.col("feature").is_in([F_S1, F_S5, F_EXT])).pivot(
            on="feature", index=["sku", "category", "window_end", "n_window"], values="raw_value"
        )
        if wide.is_empty() or not {F_S1, F_S5} <= set(wide.columns):
            return pl.DataFrame(schema=_SCHEMA)
        parts = []
        for part in wide.sort("window_end").partition_by("sku", maintain_order=True):
            s1, s5, ext, n = (
                _feature_values(part, F_S1),
                _feature_values(part, F_S5),
                _feature_values(part, F_EXT),
                part["n_window"].cast(pl.Float64).to_numpy(),
            )
            with np.errstate(invalid="ignore"):
                polar = (s1 >= p_min) & (s5 >= p_min) & (n >= min_n)  # nan -> False
                if zk is not None:
                    polar &= rolling_robust_z(ext, scale_floor=0.02) > float(zk)
            fired, _ = m_of_n(
                polar.astype(float), TriggerRule(k=0.5, m=rule.m, n=rule.n, min_valid=1)
            )
            strength = np.where(fired, np.clip(np.fmin(s1, s5) / (2 * p_min), 0.0, 1.0), 0.0)
            parts.append(
                pl.DataFrame(
                    {
                        "sku": part["sku"][0],
                        "category": part["category"][0],
                        "window_end": part["window_end"],
                        "triggered": fired,
                        "strength": strength,
                    },
                    schema=_SCHEMA,
                )
            )
        return pl.concat(parts)


def _feature_values(part: pl.DataFrame, feature: str) -> np.ndarray:
    if feature not in part.columns:
        return np.full(part.height, np.nan)
    return part[feature].cast(pl.Float64).to_numpy()
