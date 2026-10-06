"""Khuôn chung của chỉ báo. P2 (volume, rating) và P3 (content) cùng cắm vào khuôn này.

Một chỉ báo nhận bảng `feature_series` (đã lọc đúng chỉ báo của nó) và trả về các cột:
    sku, category, window_end, triggered (bool), strength (float, có thể null)
`run()` tự thêm `indicator_id`, ép kiểu và kiểm tra theo hợp đồng `indicator_matrix`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

import polars as pl

from brandsentinel.core.config import Config, IndicatorCfg
from brandsentinel.core.schemas import DataContractError, validate
from brandsentinel.core.types import Group, Table

# TODO(P2): thay bằng lớp Baseline thật của detection/baseline.py (median/MAD, partial pooling)
Baseline = Any

RESULT_COLUMNS = ["sku", "category", "window_end", "triggered", "strength"]


class Indicator(ABC):
    id: ClassVar[str]
    group: ClassVar[Group]

    def __init__(self, spec: IndicatorCfg, config: Config):
        self.spec = spec
        self.config = config

    @property
    def trigger(self):
        """Luật kích hoạt {direction, k, m, n} từ indicators.yaml."""
        return self.spec.trigger

    @property
    def params(self) -> dict[str, Any]:
        return self.spec.params

    @abstractmethod
    def compute(self, feats: pl.DataFrame, baseline: Baseline) -> pl.DataFrame:
        """`feats`: các dòng feature_series của chỉ báo này (cột: sku, category, window_end,
        indicator_id, feature, raw_value, n_window). Trả về các cột trong RESULT_COLUMNS."""

    # ---- các hàm dùng chung ----
    def select_features(self, feature_series: pl.DataFrame) -> pl.DataFrame:
        return feature_series.filter(pl.col("indicator_id") == self.id)

    @classmethod
    def empty_result(cls) -> pl.DataFrame:
        return pl.DataFrame(
            schema={
                "sku": pl.String,
                "category": pl.String,
                "window_end": pl.Date,
                "triggered": pl.Boolean,
                "strength": pl.Float64,
            }
        )

    def finalize(self, out: pl.DataFrame) -> pl.DataFrame:
        missing = [c for c in RESULT_COLUMNS if c not in out.columns]
        if missing:
            raise DataContractError(
                Table.INDICATOR_MATRIX, [f"{self.id}.compute thiếu cột {missing}"]
            )
        out = out.select(
            pl.col("sku").cast(pl.String),
            pl.col("category").cast(pl.String),
            pl.col("window_end").cast(pl.Date),
            pl.lit(self.id).alias("indicator_id"),
            pl.col("triggered").cast(pl.Boolean),
            pl.col("strength").cast(pl.Float64),
        )
        return validate(Table.INDICATOR_MATRIX, out)

    def run(self, feature_series: pl.DataFrame, baseline: Baseline) -> pl.DataFrame:
        """Điểm vào chuẩn: lọc đặc trưng, tính, kiểm tra hợp đồng."""
        return self.finalize(self.compute(self.select_features(feature_series), baseline))
