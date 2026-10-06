from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

MAD_SCALE = 1.4826  # MAD * 1.4826 ~ sigma neu du lieu gan chuan


@dataclass(frozen=True)
class Baseline:
    median: float
    scale: float  # da nhan MAD_SCALE va da chan duoi (> 0)
    n: int  # so diem hop le dung de fit


def fit_baseline(values, min_n: int = 14, scale_floor: float = 1e-6) -> Baseline | None:
    """Fit median/MAD tren cac diem huu han. Tra None neu qua it diem."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < min_n:
        return None
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med))) * MAD_SCALE
    return Baseline(med, max(mad, scale_floor), int(x.size))


def shrink_to_industry(sku_b: Baseline | None, ind_b: Baseline | None, n0: int = 30) -> Baseline | None:
    """Partial pooling: w = n/(n+n0). SKU nhieu du lieu -> tin SKU; it du lieu -> ve nganh."""
    if sku_b is None:
        return ind_b
    if ind_b is None:
        return sku_b
    w = sku_b.n / (sku_b.n + n0)
    return Baseline(
        median=w * sku_b.median + (1 - w) * ind_b.median,
        scale=w * sku_b.scale + (1 - w) * ind_b.scale,
        n=sku_b.n,
    )


def robust_z(x: float, b: Baseline | None) -> float:
    if b is None or not np.isfinite(x):
        return float("nan")
    return (x - b.median) / b.scale


def rolling_robust_z(
    values,
    lookback: int = 90,
    gap: int = 7,
    min_n: int = 14,
    industry: Baseline | None = None,
    n0: int = 30,
    scale_floor: float = 1e-6,
) -> np.ndarray:
    """z cua values[t] so voi baseline fit tren values[t-gap-lookback+1 : t-gap+1].

    gap=7: baseline chi gom cac cua so KET THUC truoc khi cua so hien tai bat dau
    (khong chong lan voi cua so dang xet). Diem khong co baseline -> nan.
    """
    x = np.asarray(values, dtype=float)
    out = np.full(x.shape, np.nan)
    for t in range(len(x)):
        hi = t - gap
        if hi < 0:
            continue
        lo = max(0, hi - lookback + 1)
        b = shrink_to_industry(fit_baseline(x[lo : hi + 1], min_n, scale_floor), industry, n0)
        out[t] = robust_z(x[t], b)
    return out


def add_robust_z(fs: pl.DataFrame, **kw) -> pl.DataFrame:
    """Them cot `z` cho bang feature_series (long). nan -> null.

    Gia dinh: moi (sku, indicator_id, feature) la chuoi ngay lien tuc (daily_agg da dien ngay thieu).
    """
    parts = []
    for part in fs.sort("window_end").partition_by(["sku", "indicator_id", "feature"], maintain_order=True):
        v = part["raw_value"].cast(pl.Float64).to_numpy()  # null -> nan
        parts.append(part.with_columns(pl.Series("z", rolling_robust_z(v, **kw), nan_to_null=True)))
    return pl.concat(parts) if parts else fs.with_columns(pl.lit(None, pl.Float64).alias("z"))