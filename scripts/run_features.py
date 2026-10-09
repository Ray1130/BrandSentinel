"""Production runner for Feature Generation pipeline (I1-I9, FEATURE_SERIES).

Đọc các bảng đầu vào (clean_reviews, daily_agg, nlp_features) theo category,
tính toán toàn bộ chuỗi đặc trưng cửa sổ trượt:
- I1-I3: Volume features
- I4-I6: Rating features
- I7-I9: Content & NLP features (term_burst_score, risk_hits, risk_distinct_users, aspect_neg_*)
Validate và lưu bảng `feature_series` dưới định dạng partitioned Parquet theo hợp đồng dữ liệu.

Sử dụng:
    python scripts/run_features.py --category dev
    python scripts/run_features.py --category Baby_Products --window-days 7
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from brandsentinel.core.config import Config, load_config
from brandsentinel.core.io import read_table, table_dir, write_table
from brandsentinel.core.types import Table
from brandsentinel.features import make_feature_series

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("brandsentinel.features_runner")


def resolve_categories(category_arg: str, cfg: Config) -> list[str]:
    """Phân giải tên alias category (dev, holdout) sang danh sách category Amazon cụ thể."""
    cats_cfg = cfg.default.data.categories
    if hasattr(cats_cfg, category_arg):
        cats = getattr(cats_cfg, category_arg)
        return list(cats) if isinstance(cats, (list, tuple)) else [str(cats)]
    return [category_arg]


def compute_feature_series_report(
    clean_df: pl.DataFrame | None,
    daily_df: pl.DataFrame | None,
    nlp_df: pl.DataFrame | None,
    feats_df: pl.DataFrame,
    runtime_seconds: float,
) -> dict[str, Any]:
    """Tổng hợp báo cáo chất lượng cho bảng feature_series."""
    n_feats = feats_df.height
    n_skus = feats_df["sku"].n_unique() if n_feats > 0 else 0
    ind_counts = dict(feats_df["indicator_id"].value_counts().iter_rows()) if n_feats > 0 else {}

    # Thống kê riêng các đặc trưng nội dung I7-I9
    i7_rows = feats_df.filter(pl.col("indicator_id") == "I7")
    i8_rows = feats_df.filter(pl.col("indicator_id") == "I8")
    i9_rows = feats_df.filter(pl.col("indicator_id") == "I9")

    i8_hits_sum = float(
        i8_rows.filter(pl.col("feature") == "risk_hits")["raw_value"].drop_nulls().sum()
    ) if not i8_rows.is_empty() else 0.0

    return {
        "inputs": {
            "clean_reviews_rows": clean_df.height if clean_df is not None else 0,
            "daily_agg_rows": daily_df.height if daily_df is not None else 0,
            "nlp_features_rows": nlp_df.height if nlp_df is not None else 0,
        },
        "output": {
            "total_feature_rows": n_feats,
            "unique_skus": n_skus,
            "indicators_distribution": ind_counts,
            "i7_burst_windows": i7_rows.height,
            "i8_total_risk_hits_in_windows": i8_hits_sum,
            "i9_aspect_windows": i9_rows.height,
            "runtime_seconds": runtime_seconds,
        },
    }


def print_feature_report(report: dict[str, Any], category: str) -> None:
    """In bảng tổng kết chất lượng bảng feature_series trực quan ra console."""
    inp = report["inputs"]
    out = report["output"]

    print("\n" + "=" * 65)
    print(f"📊 BÁO CÁO TỔNG HỢP CHUỖI ĐẶC TRƯNG FEATURE_SERIES — {category}")
    print("=" * 65)
    print("📥 NGUỒN DỮ LIỆU ĐẦU VÀO:")
    print(f"  • clean_reviews:             {inp['clean_reviews_rows']} dòng")
    print(f"  • daily_agg:                 {inp['daily_agg_rows']} dòng")
    print(f"  • nlp_features:              {inp['nlp_features_rows']} dòng")

    print("\n📤 DỮ LIỆU ĐẦU RA (feature_series):")
    print(f"  • Tổng số dòng đặc trưng:    {out['total_feature_rows']}")
    print(f"  • Số lượng SKU:              {out['unique_skus']}")
    print("  • Phân bổ theo chỉ báo (indicator_id):")
    for ind in sorted(out["indicators_distribution"].keys()):
        count = out["indicators_distribution"][ind]
        print(f"      - {ind:<5}: {count} dòng")
    print(f"  • Tổng số risk hits trong các cửa sổ (I8): {out['i8_total_risk_hits_in_windows']:.0f}")
    print(f"  • Thời gian tính toán:       {out['runtime_seconds']:.2f}s")
    print("=" * 65 + "\n")


def run_features_for_category(
    category: str,
    cfg: Config,
    *,
    window_days: int = 7,
    start: date | None = None,
    end: date | None = None,
    write: bool = True,
) -> tuple[pl.DataFrame | None, list[Path], dict[str, Any] | None]:
    """Thực thi pipeline tạo feature_series cho một category cụ thể."""
    base_clean = table_dir(cfg, Table.CLEAN_REVIEWS) / f"category={category}"
    base_daily = table_dir(cfg, Table.DAILY_AGG) / f"category={category}"
    base_nlp = table_dir(cfg, Table.NLP_FEATURES) / f"category={category}"

    log.info("Kiểm tra sự tồn tại của dữ liệu cho category '%s'...", category)

    # 1. Đọc clean_reviews
    try:
        clean_df = read_table(Table.CLEAN_REVIEWS, cfg, category=category, start=start, end=end, check=True)
    except FileNotFoundError:
        print("\n" + "=" * 65)
        print("⛔ BLOCKED — clean_reviews not available")
        print("=" * 65)
        print(f"Category mục tiêu:     {category}")
        print(f"Đường dẫn dự kiến:     {base_clean}")
        print("Lý do:                 P1 chưa kết xuất dữ liệu clean_reviews thật.")
        print("Hành động tiếp theo:   Chờ P1 cung cấp clean_reviews trước khi chạy pipeline.")
        print("=" * 65 + "\n")
        return None, [], None

    # 2. Đọc daily_agg (tùy chọn hoặc khuyến nghị)
    try:
        daily_df = read_table(Table.DAILY_AGG, cfg, category=category, start=start, end=end, check=True)
    except FileNotFoundError:
        daily_df = None
        log.warning("Bảng daily_agg chưa có tại %s. Sẽ tạo chuỗi ngày từ clean_reviews.", base_daily)

    # 3. Đọc nlp_features
    try:
        nlp_df = read_table(Table.NLP_FEATURES, cfg, category=category, start=start, end=end, check=True)
    except FileNotFoundError:
        print("\n" + "=" * 65)
        print("⛔ BLOCKED — nlp_features not available")
        print("=" * 65)
        print(f"Category mục tiêu:     {category}")
        print(f"Đường dẫn dự kiến:     {base_nlp}")
        print("Lý do:                 Bảng nlp_features chưa được tính toán.")
        print("Hành động tiếp theo:   Chạy 'python scripts/run_nlp.py --category dev' trước.")
        print("=" * 65 + "\n")
        return None, [], None

    log.info("Bắt đầu tính toán toàn bộ feature_series (I1-I9) cho category '%s'...", category)
    t0 = time.perf_counter()
    feats_df = make_feature_series(
        daily_agg=daily_df,
        clean_reviews=clean_df,
        nlp_features=nlp_df,
        window_days=window_days,
    )
    elapsed = time.perf_counter() - t0

    report = compute_feature_series_report(clean_df, daily_df, nlp_df, feats_df, elapsed)
    print_feature_report(report, category)

    written_paths: list[Path] = []
    if write and not feats_df.is_empty():
        written_paths = write_table(feats_df, Table.FEATURE_SERIES, cfg, check=True)
        log.info(
            "Đã ghi %d phân vùng parquet vào feature_series: %s",
            len(written_paths),
            [p.name for p in written_paths],
        )

    return feats_df, written_paths, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Production runner for BrandSentinel Feature Series pipeline (I1-I9).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--category",
        default="dev",
        help="Category cần chạy: 'dev', 'holdout', hoặc tên cụ thể như 'Baby_Products'",
    )
    parser.add_argument(
        "--window-days",
        type=int,
        default=7,
        help="Kích thước cửa sổ trượt (ngày)",
    )
    parser.add_argument(
        "--start",
        type=lambda s: date.fromisoformat(s),
        default=None,
        help="Ngày bắt đầu lọc YYYY-MM-DD (tùy chọn)",
    )
    parser.add_argument(
        "--end",
        type=lambda s: date.fromisoformat(s),
        default=None,
        help="Ngày kết thúc lọc YYYY-MM-DD (tùy chọn)",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Chỉ tính toán và in báo cáo, không ghi parquet ra đĩa",
    )
    parser.add_argument(
        "--config-dir",
        default=None,
        help="Đường dẫn thư mục configs tùy chỉnh (mặc định: configs/)",
    )

    args = parser.parse_args(argv)

    cfg_dir = Path(args.config_dir) if args.config_dir else (REPO_ROOT / "configs")
    cfg = load_config(cfg_dir)

    categories = resolve_categories(args.category, cfg)
    log.info("Các category sẽ được xử lý: %s", categories)

    all_written: list[Path] = []
    for cat in categories:
        _, written, _ = run_features_for_category(
            category=cat,
            cfg=cfg,
            window_days=args.window_days,
            start=args.start,
            end=args.end,
            write=not args.no_write,
        )
        all_written.extend(written)

    log.info("Hoàn tất pipeline feature_series. Tổng số file ghi: %d.", len(all_written))
    return 0


if __name__ == "__main__":
    sys.exit(main())
