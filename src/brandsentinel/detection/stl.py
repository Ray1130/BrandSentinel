"""Khu mua vu tuan bang STL tren log1p(daily_count) -> chuoi da khu mua vu cho I1.

Thiet ke (rut ra tu test tren mock):
- KHONG fit STL tren cua so ket thuc o ngay hien tai: loess/trend se "nuot" dot tang volume dang dien ra
  (phan du o diem cuoi gan nhu bang 0). Thay vao do: hoc ho so mua vu (7 gia tri theo thu) bang STL
  tren giai doan BASELINE (ket thuc truoc cua so hien tai), roi tru ho so do khoi cac ngay moi.
- Chuoi ngan (< min_cycles * period): khong du de tach mua vu -> fallback ho so mua vu nganh
  (neu co) hoac khong tru gi (zeros).
"""
from __future__ import annotations

import numpy as np
from statsmodels.tsa.seasonal import STL

from .baseline import Baseline, rolling_robust_z


def stl_residual(y, period: int = 7, min_cycles: int = 2, robust: bool = True,
                 dow=None, dow_profile=None) -> tuple[np.ndarray, str]:
    """Phan du STL cua chinh chuoi y. method in {"stl", "dow_profile", "median_only"}."""
    y = np.asarray(y, dtype=float)
    if y.size == 0:
        return y, "median_only"
    if y.size >= min_cycles * period and np.isfinite(y).all():
        return np.asarray(STL(y, period=period, robust=robust).fit().resid), "stl"
    base = y - np.nanmedian(y)
    if dow is not None and dow_profile is not None:
        return base - np.asarray(dow_profile, dtype=float)[np.asarray(dow, dtype=int)], "dow_profile"
    return base, "median_only"


def seasonal_profile(y, dow, period: int = 7, min_cycles: int = 2,
                     dow_profile=None) -> tuple[np.ndarray, str]:
    """Ho so mua vu `period` gia tri (canh giua ve 0). method in {"stl", "dow_profile", "none"}."""
    y, dow = np.asarray(y, dtype=float), np.asarray(dow, dtype=int)
    if y.size >= min_cycles * period and np.isfinite(y).all():
        seas = np.asarray(STL(y, period=period, robust=True).fit().seasonal)
        prof = np.array([seas[dow == d].mean() if (dow == d).any() else 0.0 for d in range(period)])
        return prof - prof.mean(), "stl"
    if dow_profile is not None:
        p = np.asarray(dow_profile, dtype=float)
        return p - p.mean(), "dow_profile"
    return np.zeros(period), "none"


def deseasonalize(y_log, dow=None, period: int = 7, lookback: int = 90, gap: int = 7,
                  refit_every: int = 7, dow_profile=None) -> tuple[np.ndarray, list[str]]:
    """d_t = y_t - profile[dow_t]; profile hoc tu y[t-gap-lookback+1 : t-gap+1] (khong ro ri tuong lai)."""
    y = np.asarray(y_log, dtype=float)
    dow = np.arange(len(y)) % period if dow is None else np.asarray(dow, dtype=int)
    out, methods, prof, m = np.empty_like(y), [], np.zeros(period), "none"
    for t in range(len(y)):
        if t % refit_every == 0:
            hi, lo = t - gap, max(0, t - gap - lookback + 1)
            prof, m = seasonal_profile(y[lo : hi + 1], dow[lo : hi + 1], period, dow_profile=dow_profile) \
                if hi >= lo else seasonal_profile([], [], period, dow_profile=dow_profile)
        out[t] = y[t] - prof[dow[t]]
        methods.append(m)
    return out, methods


def trailing_mean(x, w: int = 7) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.full(x.shape, np.nan)
    for t in range(w - 1, len(x)):
        seg = x[t - w + 1 : t + 1]
        if np.isfinite(seg).all():
            out[t] = seg.mean()
    return out


def volume_surge_z(counts, dow=None, period: int = 7, lookback: int = 90, gap: int = 7,
                   dow_profile=None, industry: Baseline | None = None,
                   scale_floor: float = 0.05) -> tuple[np.ndarray, list[str]]:
    """I1: log1p(count) -> tru mua vu (STL tren baseline) -> trung binh 7 ngay -> robust z so voi baseline.

    scale_floor=0.05 (~5% thay doi volume tren thang log): chan duoi do lech chuan de SKU rat on dinh
    khong sinh z phong dai. Chinh trong indicators.yaml.
    """
    d, methods = deseasonalize(np.log1p(np.asarray(counts, float)), dow, period, lookback, gap,
                               dow_profile=dow_profile)
    z = rolling_robust_z(trailing_mean(d, period), lookback=lookback, gap=gap,
                         industry=industry, scale_floor=scale_floor)
    return z, methods