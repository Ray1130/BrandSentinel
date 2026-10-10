"""Đọc/ghi các bảng của hợp đồng dữ liệu dưới dạng Parquet.

Bố cục:  <zone>/<table>/category=<category>/<yyyy-mm>.parquet
  zone `interim`  : clean_reviews, daily_agg, nlp_features
  zone `processed`: feature_series, indicator_matrix, alerts, recommendations/
  zone `interim`  : recall_labels (verified external labels, partitioned by recall_date)

Ghi là IDEMPOTENT theo (category, tháng, SKU, khoảng ngày): dòng cũ cùng SKU nằm trong khoảng
ngày [min, max] của dữ liệu mới bị thay thế, các dòng khác được giữ nguyên. Bảng không có cột
`sku` (nlp_features) được thay theo khóa. Chạy lại cùng một stage không nhân đôi dữ liệu.
"""

from __future__ import annotations

import json
import os
import re
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

from brandsentinel.core.config import Config
from brandsentinel.core.schemas import DATE_COLUMN, KEYS, validate
from brandsentinel.core.types import Table

ZONE: dict[Table, str] = {
    Table.CLEAN_REVIEWS: "data_interim",
    Table.DAILY_AGG: "data_interim",
    Table.NLP_FEATURES: "data_interim",
    Table.FEATURE_SERIES: "data_processed",
    Table.INDICATOR_MATRIX: "data_processed",
    Table.ALERTS: "data_processed",
    Table.RECALL_LABELS: "data_interim",
}
_SAFE = re.compile(r"[A-Za-z0-9_.\-]+")


def _safe(name: str) -> str:
    """Tên category dùng làm tên thư mục: chặn ký tự không hợp lệ trên Windows (: / \\ ...)."""
    if not _SAFE.fullmatch(name):
        raise ValueError(f"tên không dùng được làm tên thư mục: {name!r}")
    return name


def table_dir(cfg: Config, table: Table) -> Path:
    return cfg.path(ZONE[table]) / table.value  # type: ignore[arg-type]


def partition_path(cfg: Config, table: Table, category: str, month: str) -> Path:
    return table_dir(cfg, table) / f"category={_safe(category)}" / f"{month}.parquet"


def write_table(df: pl.DataFrame, table: Table, cfg: Config, *, check: bool = True) -> list[Path]:
    """Ghi `df` vào các phân vùng (category, tháng); trả danh sách file đã ghi."""
    if check:
        validate(table, df)
    if df.is_empty():
        return []
    date_col, keys = DATE_COLUMN[table], KEYS[table]
    tagged = df.with_columns(pl.col(date_col).dt.strftime("%Y-%m").alias("_month"))
    written: list[Path] = []
    for (category, month), chunk in tagged.partition_by(
        ["category", "_month"], as_dict=True
    ).items():
        chunk = chunk.drop("_month")
        path = partition_path(cfg, table, str(category), str(month))
        if path.exists():
            old = pl.read_parquet(path)
            if "sku" in chunk.columns:
                lo, hi = chunk[date_col].min(), chunk[date_col].max()
                replaced = pl.col(date_col).is_between(lo, hi) & pl.col("sku").is_in(
                    chunk["sku"].unique().to_list()
                )
                old = old.filter(~replaced)
            else:
                old = old.join(chunk.select(keys), on=keys, how="anti")
            chunk = pl.concat([old, chunk], how="vertical")
        chunk = chunk.sort(keys)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        chunk.write_parquet(tmp, compression=cfg.default.paths.parquet_compression)
        os.replace(tmp, path)  # ghi nguyên tử: không để lại file dở khi lỗi giữa chừng
        written.append(path)
    return written


def read_table(
    table: Table,
    cfg: Config,
    *,
    category: str | None = None,
    start: date | None = None,
    end: date | None = None,
    skus: list[str] | None = None,
    check: bool = False,
) -> pl.DataFrame:
    """Đọc bảng, lọc theo category / khoảng ngày [start, end] / danh sách SKU."""
    base = table_dir(cfg, table)
    pattern = f"category={_safe(category)}" if category else "category=*"
    files = sorted(base.glob(f"{pattern}/*.parquet"))
    if not files:
        raise FileNotFoundError(f"chưa có dữ liệu cho {table.value} (category={category}) ở {base}")
    lf = pl.concat([pl.scan_parquet(p) for p in files], how="vertical")
    date_col = DATE_COLUMN[table]
    if start is not None:
        lf = lf.filter(pl.col(date_col) >= start)
    if end is not None:
        lf = lf.filter(pl.col(date_col) <= end)
    if skus is not None:
        lf = lf.filter(pl.col("sku").is_in(skus))
    df = lf.sort(KEYS[table]).collect()
    return validate(table, df) if check else df


# ------------------------- khuyến nghị LLM (JSON) -------------------------
def recommendation_path(cfg: Config, category: str, sku: str, window_end: date) -> Path:
    name = f"{_safe(sku)}_{window_end.isoformat()}.json"
    return cfg.path("data_processed") / "recommendations" / f"category={_safe(category)}" / name


def recommendation_cache_path(cfg: Config, evidence_hash: str) -> Path:
    """Return the stable recommendation path for a canonical evidence-package SHA-256."""
    if not re.fullmatch(r"[0-9a-f]{64}", evidence_hash):
        raise ValueError("evidence_hash phải là SHA-256 dạng hex thường")
    return cfg.path("data_processed") / "recommendations" / "cache" / f"{evidence_hash}.json"


def write_json(path: Path, obj: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))
