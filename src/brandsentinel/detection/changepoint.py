"""PELT (ruptures) dung chung. I5 dung lam dieu kien kich hoat; I1/I3/I4 chi dung lam bang chung bo sung.

Ba rang buoc theo thiet ke:
- PELT la offline -> chi chay tren `max_window_days` ngay cuoi, va chi chap nhan diem gay trong `accept_recent_days` cuoi.
- Chuan hoa chuoi bang sigma robust truoc khi fit nen penalty khong phu thuoc don vi: pen = pen_scale * log(n).
- Loc theo do lech trung binh >= min_effect_sigma (tranh diem gay nho nhung "co y nghia thong ke").
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import ruptures as rpt


@dataclass(frozen=True)
class ChangepointConfig:
    model: str = "l2"  # dich trung binh tren chuoi da chuan hoa
    min_size: int = 5
    jump: int = 1
    pen_scale: float = 3.0
    max_window_days: int = 90
    accept_recent_days: int = 14
    min_effect_sigma: float = 1.0
    direction: Literal["up", "down", "both"] = "both"


def robust_sigma(x) -> float:
    """sigma robust tu sai phan bac 1 (khong bi keo boi dich muc): MAD(diff)/sqrt(2)."""
    x = np.asarray(x, dtype=float)
    d = np.diff(x[np.isfinite(x)])
    if d.size < 3:
        return float("nan")
    return float(1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2))


def detect_changepoints(series, *, cfg: ChangepointConfig = ChangepointConfig(),
                        sigma: float | None = None) -> list[int]:
    """Tra vi tri (chi so trong `series` goc) bat dau doan moi, da loc theo cfg. Khong co -> [].

    Luu y: sigma uoc luong tu sai phan se THAP neu chuoi tu tuong quan manh (cua so truot chong lan 6/7);
    voi chuoi nhu vay nen truyen `sigma` tu baseline.
    """
    x = np.asarray(series, dtype=float)
    offset = max(0, len(x) - cfg.max_window_days)
    seg = x[offset:]
    finite = np.isfinite(seg)
    if finite.sum() < 2 * cfg.min_size:
        return []
    sg = sigma if sigma is not None else robust_sigma(seg)
    if not np.isfinite(sg) or sg <= 0:
        return []
    z = (np.where(finite, seg, np.median(seg[finite])) - np.median(seg[finite])) / sg
    bkps = rpt.Pelt(model=cfg.model, min_size=cfg.min_size, jump=cfg.jump).fit(z).predict(
        pen=cfg.pen_scale * np.log(len(z)))
    edges = [0, *bkps]  # bkps ket thuc bang len(z)
    out = []
    for i in range(1, len(edges) - 1):
        cp = edges[i]
        if cp < len(z) - cfg.accept_recent_days:
            continue
        effect = z[cp : edges[i + 1]].mean() - z[edges[i - 1] : cp].mean()
        ok = {"up": effect, "down": -effect, "both": abs(effect)}[cfg.direction] >= cfg.min_effect_sigma
        if ok:
            out.append(offset + cp)
    return out