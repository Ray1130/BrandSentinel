"""Stage detect: feature_series -> indicator_matrix (moi chi bao tu `Indicator.run`, roi gop)."""

from __future__ import annotations

import polars as pl

from ..core.registry import build_indicators
from . import indicators  # noqa: F401  (import de @register chay)


def detect(feature_series: pl.DataFrame, cfg, baseline=None) -> pl.DataFrame:
    parts = [ind.run(feature_series, baseline) for ind in build_indicators(cfg).values()]
    if not parts:
        raise ValueError("build_indicators(cfg) khong tra chi bao nao")
    matrix = pl.concat(parts).sort(["sku", "indicator_id", "window_end"])
    dup = matrix.group_by(["sku", "window_end", "indicator_id"]).len().filter(pl.col("len") > 1)
    if dup.height:
        raise ValueError(
            f"indicator_matrix trung khoa (sku, window_end, indicator_id): {dup.head(3).to_dicts()}"
        )
    return matrix
