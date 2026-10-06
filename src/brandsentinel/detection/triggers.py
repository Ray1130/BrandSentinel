from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import polars as pl
from numpy.lib.stride_tricks import sliding_window_view


@dataclass(frozen=True)
class TriggerRule:
    k: float  # nguong z
    m: int  # so cua so can vuot
    n: int  # so cua so gan nhat xet (n >= m)
    direction: Literal["up", "down", "both"] = "up"
    min_valid: int | None = None  # so diem hop le toi thieu trong n (mac dinh = m)
    feature: str | None = None  # feature chinh cua chi bao (khi feature_series co nhieu feature)


def _oriented(z: np.ndarray, direction: str) -> np.ndarray:
    if direction == "up":
        return z
    if direction == "down":
        return -z
    return np.abs(z)


def m_of_n(z, rule: TriggerRule) -> tuple[np.ndarray, np.ndarray]:
    """Tra (triggered: bool[T], strength: float[T] trong [0,1]). nan khong tinh la vuot nguong."""
    z = np.asarray(z, dtype=float)
    s = _oriented(z, rule.direction)
    valid = np.isfinite(s)
    exceed = valid & (np.where(valid, s, -np.inf) > rule.k)

    def trailing_sum(a: np.ndarray) -> np.ndarray:  # tong n phan tu gan nhat (thieu dau -> tong rieng phan co)
        return np.convolve(a.astype(int), np.ones(rule.n, dtype=int))[: len(a)]

    fired = (trailing_sum(exceed) >= rule.m) & (trailing_sum(valid) >= (rule.min_valid or rule.m))

    padded = np.concatenate([np.full(rule.n - 1, -np.inf), np.where(valid, s, -np.inf)])
    peak = sliding_window_view(padded, rule.n).max(axis=1)
    strength = np.where(fired, np.clip((peak - rule.k) / rule.k, 0.0, 1.0), 0.0)
    return fired, strength


def build_indicator_matrix(zdf: pl.DataFrame, rules: dict[str, TriggerRule]) -> pl.DataFrame:
    """zdf: (sku, window_end, indicator_id, feature, z) -> indicator_matrix (sku, window_end, indicator_id, triggered, strength)."""
    parts = []
    for ind_id, rule in rules.items():
        sub = zdf.filter(pl.col("indicator_id") == ind_id)
        if rule.feature is not None:
            sub = sub.filter(pl.col("feature") == rule.feature)
        for part in sub.sort("window_end").partition_by("sku", maintain_order=True):
            fired, strength = m_of_n(part["z"].cast(pl.Float64).to_numpy(), rule)
            parts.append(
                part.select("sku", "window_end", "indicator_id").with_columns(
                    pl.Series("triggered", fired), pl.Series("strength", strength)
                )
            )
    empty = pl.DataFrame(schema={"sku": pl.Utf8, "window_end": pl.Date, "indicator_id": pl.Utf8,
                                 "triggered": pl.Boolean, "strength": pl.Float64})
    return pl.concat(parts) if parts else empty