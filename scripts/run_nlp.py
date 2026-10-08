"""Production runner for NLP Quality pipeline (I8, I9, I10, NLP_FEATURES).

Đọc bảng `clean_reviews` thật từ P1, chạy toàn bộ pipeline NLP:
- Sentiment & mismatch (Task 2)
- Risk Layer 1 (Lexicon) + Layer 2 (Embedding) (Task 3 + Phase 2)
- Aspect & aspect sentiment (Task 4)
Validate và lưu bảng `nlp_features` dưới định dạng partitioned Parquet theo hợp đồng dữ liệu.

Sử dụng:
    python scripts/run_nlp.py --category dev
    python scripts/run_nlp.py --category Baby_Products --start 2023-01-01 --end 2023-03-31
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
from brandsentinel.features.nlp.pipeline import compute_nlp_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("brandsentinel.nlp_runner")


def resolve_categories(category_arg: str, cfg: Config) -> list[str]:
    """Phân giải tên alias category (dev, holdout) sang danh sách category Amazon cụ thể."""
    cats_cfg = cfg.default.data.categories
    if hasattr(cats_cfg, category_arg):
        cats = getattr(cats_cfg, category_arg)
        return list(cats) if isinstance(cats, (list, tuple)) else [str(cats)]
    return [category_arg]


def compute_data_quality_report(
    clean_df: pl.DataFrame,
    nlp_df: pl.DataFrame,
    runtime_seconds: float,
) -> dict[str, Any]:
    """Tổng hợp báo cáo chất lượng dữ liệu đầu vào và đầu ra của pipeline NLP."""
    n_in = clean_df.height
    n_out = nlp_df.height
    unique_in = clean_df["review_id"].n_unique() if n_in > 0 else 0
    unique_out = nlp_df["review_id"].n_unique() if n_out > 0 else 0

    # Kiểm tra null / empty text_norm
    if "text_norm" in clean_df.columns:
        null_text = int(clean_df["text_norm"].null_count())
        empty_text = int(
            clean_df.filter(pl.col("text_norm").is_not_null() & (pl.col("text_norm").str.strip_chars() == "")).height
        )
    else:
        null_text = n_in
        empty_text = 0

    null_text_rate = (null_text / n_in) if n_in > 0 else 0.0
    empty_text_rate = (empty_text / n_in) if n_in > 0 else 0.0

    # Thống kê ngày
    date_col = clean_df["date"] if "date" in clean_df.columns else None
    min_date = date_col.min().isoformat() if (date_col is not None and n_in > 0 and date_col.min() is not None) else "N/A"
    max_date = date_col.max().isoformat() if (date_col is not None and n_in > 0 and date_col.max() is not None) else "N/A"

    # Thống kê đầu ra NLP
    risk_hits = int(nlp_df["risk_hit"].sum()) if n_out > 0 else 0
    risk_hit_rate = (risk_hits / n_out) if n_out > 0 else 0.0

    mismatches = int(nlp_df["mismatch"].sum()) if (n_out > 0 and nlp_df["mismatch"].is_not_null().any()) else 0
    mismatch_rate = (mismatches / n_out) if n_out > 0 else 0.0

    aspect_dist: dict[str, int] = {}
    if n_out > 0:
        for val, cnt in nlp_df["aspect"].value_counts().iter_rows():
            aspect_dist[str(val) if val is not None else "null"] = int(cnt)
    null_aspects = aspect_dist.get("null", 0)

    throughput = (n_out / runtime_seconds) if runtime_seconds > 0 else 0.0

    return {
        "input": {
            "rows": n_in,
            "unique_review_ids": unique_in,
            "date_range": f"{min_date} .. {max_date}",
            "null_text_norm_count": null_text,
            "null_text_norm_rate": null_text_rate,
            "empty_text_norm_count": empty_text,
            "empty_text_norm_rate": empty_text_rate,
        },
        "output": {
            "rows": n_out,
            "unique_review_ids": unique_out,
            "risk_hit_count": risk_hits,
            "risk_hit_rate": risk_hit_rate,
            "mismatch_count": mismatches,
            "mismatch_rate": mismatch_rate,
            "aspect_distribution": aspect_dist,
            "null_aspect_count": null_aspects,
            "runtime_seconds": runtime_seconds,
            "throughput_reviews_per_sec": throughput,
        },
    }


def print_summary_report(report: dict[str, Any], category: str) -> None:
    """In bảng tổng kết chất lượng dữ liệu trực quan ra console."""
    inp = report["input"]
    out = report["output"]

    print("\n" + "=" * 60)
    print(f"📊 BÁO CÁO CHẤT LƯỢNG PIPELINE NLP — CATEGORY: {category}")
    print("=" * 60)
    print("📥 DỮ LIỆU ĐẦU VÀO (clean_reviews):")
    print(f"  • Tổng số dòng:              {inp['rows']}")
    print(f"  • Unique review_id:          {inp['unique_review_ids']}")
    print(f"  • Khoảng thời gian:          {inp['date_range']}")
    print(f"  • Tỷ lệ text_norm null:      {inp['null_text_norm_count']} ({inp['null_text_norm_rate']:.2%})")
    print(f"  • Tỷ lệ text_norm rỗng:      {inp['empty_text_norm_count']} ({inp['empty_text_norm_rate']:.2%})")

    print("\n📤 DỮ LIỆU ĐẦU RA (nlp_features):")
    print(f"  • Tổng số dòng:              {out['rows']}")
    print(f"  • Unique review_id:          {out['unique_review_ids']}")
    print(f"  • Cảnh báo rủi ro (risk_hit): {out['risk_hit_count']} ({out['risk_hit_rate']:.2%})")
    print(f"  • Mâu thuẫn sao (mismatch):  {out['mismatch_count']} ({out['mismatch_rate']:.2%})")
    print("  • Phân bổ khía cạnh (aspect):")
    for asp, count in out["aspect_distribution"].items():
        pct = (count / out["rows"]) if out["rows"] > 0 else 0.0
        print(f"      - {asp:<10}: {count} ({pct:.2%})")
    print(f"  • Thời gian xử lý:           {out['runtime_seconds']:.2f}s")
    print(f"  • Tốc độ thông lượng:        {out['throughput_reviews_per_sec']:.1f} reviews/giây")
    print("=" * 60 + "\n")


def run_nlp_for_category(
    category: str,
    cfg: Config,
    *,
    start: date | None = None,
    end: date | None = None,
    write: bool = True,
    risk_similarity_threshold: float | None = None,
) -> tuple[pl.DataFrame | None, list[Path], dict[str, Any] | None]:
    """Thực thi pipeline NLP cho một category cụ thể."""
    base_dir = table_dir(cfg, Table.CLEAN_REVIEWS)
    expected_cat_dir = base_dir / f"category={category}"

    log.info("Kiểm tra dữ liệu clean_reviews cho category '%s' ở %s...", category, expected_cat_dir)

    try:
        clean_df = read_table(
            Table.CLEAN_REVIEWS,
            cfg,
            category=category,
            start=start,
            end=end,
            check=True,
        )
    except FileNotFoundError:
        print("\n" + "=" * 65)
        print("⛔ BLOCKED — clean_reviews not available")
        print("=" * 65)
        print(f"Category mục tiêu:     {category}")
        print(f"Đường dẫn dự kiến:     {expected_cat_dir}")
        print("Lý do:                 P1 chưa kết xuất dữ liệu clean_reviews thật.")
        print("Hành động tiếp theo:   Chờ P1 hoàn tất giai đoạn tiền xử lý (preprocess)")
        print("                       trước khi chạy lại lệnh này.")
        print("=" * 65 + "\n")
        return None, [], None

    if clean_df.is_empty():
        log.warning("Bảng clean_reviews cho category '%s' rỗng.", category)

    t0 = time.perf_counter()
    nlp_df = compute_nlp_features(
        clean_df,
        cfg=cfg,
        risk_similarity_threshold=risk_similarity_threshold,
        validate_output=True,
    )
    elapsed = time.perf_counter() - t0

    report = compute_data_quality_report(clean_df, nlp_df, elapsed)
    print_summary_report(report, category)

    written_paths: list[Path] = []
    if write and not nlp_df.is_empty():
        written_paths = write_table(nlp_df, Table.NLP_FEATURES, cfg, check=True)
        log.info("Đã ghi %d phân vùng parquet vào nlp_features: %s", len(written_paths), [p.name for p in written_paths])

    return nlp_df, written_paths, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Production runner for BrandSentinel NLP Quality pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--category",
        default="dev",
        help="Category cần chạy: 'dev', 'holdout', hoặc tên cụ thể như 'Baby_Products'",
    )
    parser.add_argument(
        "--start",
        type=lambda s: date.fromisoformat(s),
        default=None,
        help="Ngày bắt đầu UTC (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        type=lambda s: date.fromisoformat(s),
        default=None,
        help="Ngày kết thúc UTC (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Chỉ chạy inference và validate, không ghi ra parquet",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Ghi đè ngưỡng tương đồng rủi ro Layer 2 (mặc định lấy từ indicators.yaml: 0.60)",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=None,
        help="Thư mục chứa configs (mặc định configs/)",
    )

    args = parser.parse_args(argv)

    cfg = load_config(args.config_dir)
    target_categories = resolve_categories(args.category, cfg)
    log.info("Phân giải category '%s' -> %s", args.category, target_categories)

    any_success = False
    all_blocked = True

    for cat in target_categories:
        nlp_df, written, _ = run_nlp_for_category(
            category=cat,
            cfg=cfg,
            start=args.start,
            end=args.end,
            write=not args.no_write,
            risk_similarity_threshold=args.threshold,
        )
        if nlp_df is not None:
            any_success = True
            all_blocked = False

    if all_blocked:
        return 0  # Clean exit as requested when clean_reviews is not available

    return 0 if any_success else 1


if __name__ == "__main__":
    sys.exit(main())
