"""Nhom volume: I1 (dot bien sau khu mua vu), I2 (toc do tang bat thuong), I3 (xu huong tieu cuc keo dai).

Loi tinh toan nam o `run_indicator` (ham thuan tren panel long, khong phu thuoc base.py). Ba lop Indicator
o cuoi chi la vo mong. GIA DINH ve base.py (can doi chieu voi ban that): `Indicator` co thuoc tinh lop `id`,
`group`; `self.trigger` la TriggerRule hoac mapping/obj co k, m, n; `self.params` la mapping;
`compute(feats, baseline)` tra (sku, category, window_end, triggered, strength).
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import date

import numpy as np
import polars as pl

from ..baseline import fit_baseline, rolling_robust_z
from ..base import Indicator, register
from ..stl import deseasonalize, trailing_mean
from ..trend import mk_trend
from ..triggers import TriggerRule, m_of_n

F_COUNT, F_GROWTH, F_NEG = "log1p_daily_count", "growth_rate", "neg_ratio_shrunk"


def _daily(part: pl.DataFrame) -> tuple[pl.Series, np.ndarray]:
    """Chuoi lien tuc theo ngay cua 1 SKU; ngay thieu/null -> nan (khong dien 0)."""
    p = part.sort("window_end")
    d0, d1 = p["window_end"].min(), p["window_end"].max()
    full = pl.DataFrame({"window_end": pl.date_range(d0, d1, "1d", eager=True)}).join(
        p.select("window_end", "raw_value"), on="window_end", how="left")
    return full["window_end"], full["raw_value"].cast(pl.Float64).to_numpy()


def run_indicator(
    feats: pl.DataFrame,
    feature: str,
    to_series: Callable[[pl.Series, np.ndarray], np.ndarray],
    rule: TriggerRule,
    *,
    use_baseline: bool = True,
    industry_days: int = 60,
    industry_until: date | None = None,
    z_kw: Mapping | None = None,
) -> pl.DataFrame:
    """Panel long -> (sku, category, window_end, triggered, strength) cho 1 chi bao.

    to_series: (dates, values) -> chuoi can lay z (use_baseline=True) hoac chinh la z (use_baseline=False).
    Baseline nganh chi fit tren giai doan dau (industry_until hoac industry_days diem dau) cua moi SKU
    de dot khung hoang khong lot vao baseline.
    """
    z_kw = dict(z_kw or {})
    data: dict[str, tuple[str, pl.Series, np.ndarray]] = {}
    for part in feats.filter(pl.col("feature") == feature).partition_by("sku", maintain_order=True):
        dates, vals = _daily(part)
        data[part["sku"][0]] = (part["category"][0], dates, to_series(dates, vals))

    industry: dict = {}
    if use_baseline:
        pooled: dict[str, list[np.ndarray]] = {}
        for cat, dates, s in data.values():
            mask = (dates <= industry_until).to_numpy() if industry_until else np.arange(len(s)) < industry_days
            pooled.setdefault(cat, []).append(s[mask])
        industry = {c: fit_baseline(np.concatenate(v), min_n=14, scale_floor=z_kw.get("scale_floor", 1e-6))
                    for c, v in pooled.items()}

    parts = []
    for sku, (cat, dates, s) in data.items():
        z = rolling_robust_z(s, industry=industry.get(cat), **z_kw) if use_baseline else s
        fired, strength = m_of_n(z, rule)
        parts.append(pl.DataFrame({"sku": sku, "category": cat, "window_end": dates,
                                   "triggered": fired, "strength": strength}))
    schema = {"sku": pl.Utf8, "category": pl.Utf8, "window_end": pl.Date,
              "triggered": pl.Boolean, "strength": pl.Float64}
    return pl.concat(parts) if parts else pl.DataFrame(schema=schema)


# ---- ba chi bao ------------------------------------------------------------------------------------
def _i1_series(dates: pl.Series, v: np.ndarray) -> np.ndarray:
    dow = dates.dt.weekday().to_numpy() - 1  # polars: Mon=1 -> 0
    d, _ = deseasonalize(v, dow)  # v = log1p_daily_count DA la log1p -> khong log lan 2
    return trailing_mean(d, 7)


def _mk_z(v: np.ndarray, last: int = 28, min_n: int = 10) -> np.ndarray:
    """z cua Mann-Kendall (Hamed-Rao) theo thoi gian, chi giu khi xu huong TANG. Khong tinh duoc -> nan (khong phai 0)."""
    out = np.full(len(v), np.nan)
    for t in range(len(v)):
        r = mk_trend(v[: t + 1], last=last, min_n=min_n)
        if r.trend != "insufficient" and np.isfinite(r.z):
            out[t] = r.z if (r.z > 0 and r.slope > 0) else 0.0
    return out


def _as_rule(t, **defaults) -> TriggerRule:
    if isinstance(t, TriggerRule):
        return t
    get = (lambda k: t[k]) if isinstance(t, Mapping) else (lambda k: getattr(t, k))
    return TriggerRule(k=float(get("k")), m=int(get("m")), n=int(get("n")), **defaults)


@register
class VolumeSurge(Indicator):
    id, group = "I1", "volume"

    def compute(self, feats, baseline):
        return run_indicator(feats, F_COUNT, _i1_series, _as_rule(self.trigger),
                             z_kw={"scale_floor": self.params.get("scale_floor", 0.05)})


@register
class GrowthAnomaly(Indicator):
    """I2: z cua feature `growth_rate`. Dinh nghia feature (tong 7 ngay so voi 7 ngay truoc) thuoc ve M2/P1."""
    id, group = "I2", "volume"

    def compute(self, feats, baseline):
        return run_indicator(feats, F_GROWTH, lambda dates, v: v, _as_rule(self.trigger),
                             z_kw={"scale_floor": self.params.get("scale_floor", 0.05)})


@register
class NegativeTrend(Indicator):
    """I3: Mann-Kendall (Hamed-Rao) tren neg_ratio_shrunk; z cua MK di thang vao m-of-n (k ~ 1.645)."""
    id, group = "I3", "volume"

    def compute(self, feats, baseline):
        last = int(self.params.get("mk_last", 28))
        return run_indicator(feats, F_NEG, lambda dates, v: _mk_z(v, last=last),
                             _as_rule(self.trigger), use_baseline=False)