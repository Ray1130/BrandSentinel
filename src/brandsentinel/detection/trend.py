"""I3: xu huong tieu cuc keo dai = Mann-Kendall tren ty le tieu cuc da co ve trung binh nganh (Beta)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pymannkendall as mk


def neg_ratio_shrunk(neg, n, prior_rate: float, strength: float = 20.0) -> np.ndarray:
    """(neg + a*p0) / (n + a). n nho -> ve p0 (nganh); n lon -> ty le thuc.

    Luu y: n=0 tra ve p0 (khong phai null) -> luon kem dieu kien n_window toi thieu o noi goi.
    """
    neg = np.asarray(neg, dtype=float)
    n = np.asarray(n, dtype=float)
    return (neg + strength * prior_rate) / (n + strength)


@dataclass(frozen=True)
class MKResult:
    trend: str  # "increasing" | "decreasing" | "no trend" | "insufficient"
    z: float
    p: float
    slope: float


def mk_trend(values, last: int = 28, min_n: int = 10, alpha: float = 0.05, hamed: bool = True) -> MKResult:
    """Mann-Kendall tren `last` diem gan nhat. hamed=True: hieu chinh tu tuong quan (cua so truot chong lan)."""
    x = np.asarray(values, dtype=float)[-last:]
    x = x[np.isfinite(x)]
    if x.size < min_n or np.ptp(x) == 0:
        return MKResult("insufficient" if x.size < min_n else "no trend", 0.0, 1.0, 0.0)
    try:
        r = (mk.hamed_rao_modification_test if hamed else mk.original_test)(x, alpha=alpha)
    except Exception:  # chuoi suy bien/nhieu hang so
        r = mk.original_test(x, alpha=alpha)
    if not np.isfinite(r.z):  # Hamed-Rao tra nan khi chuoi don dieu hoan toan -> dung MK goc
        r = mk.original_test(x, alpha=alpha)
    return MKResult(r.trend, float(r.z), float(r.p), float(r.slope))


def mk_z_series(values, last: int = 28, min_n: int = 10, min_slope: float = 0.0, **kw) -> np.ndarray:
    """Chuoi z cua Mann-Kendall theo thoi gian (chi giu z khi xu huong TANG va slope > min_slope).

    Dung lam dau vao cho triggers.m_of_n voi k ~ 1.645 (one-sided 5%).
    """
    x = np.asarray(values, dtype=float)
    out = np.full(x.shape, np.nan)
    for t in range(len(x)):
        r = mk_trend(x[: t + 1], last=last, min_n=min_n, **kw)
        if r.trend == "insufficient":
            continue
        out[t] = r.z if (r.z > 0 and r.slope > min_slope) else 0.0
    return out