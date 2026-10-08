"""Evaluation runner comparing Layer 1, Layer 2, and Combined I8 risk detection.

Đánh giá định lượng trên tập mẫu gán nhãn `sample_reviews.csv`:
- Layer 1 (Rule-based lexicon)
- Layer 2 (Semantic embedding similarity)
- Combined (Layer 1 OR Layer 2 >= threshold)

Xuất báo cáo chi tiết gồm Precision, Recall, F1, Confusion Matrix, và danh sách FP / FN
phục vụ tinh chỉnh từ điển và hiệu chỉnh ngưỡng (threshold calibration).

Sử dụng:
    python scripts/evaluate_risk.py
    python scripts/evaluate_risk.py --threshold 0.40 --output tests/fixtures/risk_evaluation_report.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from brandsentinel.core.config import Config, load_config
from brandsentinel.features.nlp.embedder import compute_i8_combined, get_risk_embedder
from brandsentinel.features.nlp.risk_lexicon import get_risk_detector

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("brandsentinel.evaluate_risk")


def evaluate_method(
    df: pl.DataFrame,
    pred_col: str,
    true_col: str = "expected_risk_hit",
) -> dict[str, Any]:
    """Tính toán metrics phân loại nhị phân và trích xuất danh sách FP / FN."""
    y_pred = df[pred_col].to_numpy()
    y_true = df[true_col].to_numpy()

    tp_mask = y_pred & y_true
    fp_mask = y_pred & ~y_true
    fn_mask = ~y_pred & y_true
    tn_mask = ~y_pred & ~y_true

    tp = int(np.sum(tp_mask))
    fp = int(np.sum(fp_mask))
    fn = int(np.sum(fn_mask))
    tn = int(np.sum(tn_mask))

    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    # Trích xuất danh sách các ca nhầm lẫn để phân tích
    fp_df = df.filter(pl.Series(fp_mask))
    fn_df = df.filter(pl.Series(fn_mask))

    fp_list = [
        {
            "review_id": str(r.get("review_id")),
            "text": str(r.get("text_norm", r.get("text_raw", ""))),
            "test_group": str(r.get("test_group", "")),
            "risk_sim": float(r["risk_sim"]) if r.get("risk_sim") is not None else None,
            "predicted_risk_terms": r.get("risk_terms"),
        }
        for r in fp_df.to_dicts()
    ]

    fn_list = [
        {
            "review_id": str(r.get("review_id")),
            "text": str(r.get("text_norm", r.get("text_raw", ""))),
            "test_group": str(r.get("test_group", "")),
            "risk_sim": float(r["risk_sim"]) if r.get("risk_sim") is not None else None,
            "expected_risk_terms": r.get("expected_risk_terms"),
        }
        for r in fn_df.to_dicts()
    ]

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        },
        "fp_count": len(fp_list),
        "fn_count": len(fn_list),
        "fp_samples": fp_list,
        "fn_samples": fn_list,
    }


def run_evaluation(
    input_path: Path,
    output_path: Path | None = None,
    cfg: Config | None = None,
    benchmark_threshold: float | None = None,
) -> dict[str, Any]:
    """Chạy toàn bộ quy trình so sánh Layer 1, Layer 2 và Combined trên tập gán nhãn."""
    if not input_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu mẫu: {input_path}")

    active_cfg = cfg or load_config()

    # Đọc production threshold từ indicators.yaml
    i8_spec = active_cfg.indicators.indicators.get("I8")
    prod_threshold = float(i8_spec.params.get("embedding_similarity_min", 0.60)) if i8_spec else 0.60
    bench_threshold = benchmark_threshold if benchmark_threshold is not None else 0.40

    log.info("Nạp dữ liệu ground truth từ %s...", input_path)
    df = pl.read_csv(input_path)

    detector = get_risk_detector(cfg=active_cfg)
    embedder = get_risk_embedder(cfg=active_cfg)

    # 1. Chạy Layer 1 only
    l1_res = compute_i8_combined(df, cfg=active_cfg, detector=detector, embedder=embedder, mode="layer1")
    df = df.with_columns(
        l1_res["risk_hit"].alias("l1_hit"),
        l1_res["risk_terms"].alias("risk_terms"),
        l1_res["risk_sim"].alias("risk_sim"),
    )

    # 2. Chạy Layer 2 only (Production threshold)
    l2_prod = compute_i8_combined(
        df, cfg=active_cfg, detector=detector, embedder=embedder, threshold=prod_threshold, mode="layer2"
    )
    df = df.with_columns(l2_prod["risk_hit"].alias("l2_prod_hit"))

    # 3. Chạy Combined (Production threshold)
    comb_prod = compute_i8_combined(
        df, cfg=active_cfg, detector=detector, embedder=embedder, threshold=prod_threshold, mode="combined"
    )
    df = df.with_columns(comb_prod["risk_hit"].alias("comb_prod_hit"))

    # 4. Chạy Layer 2 only (Benchmark threshold)
    l2_bench = compute_i8_combined(
        df, cfg=active_cfg, detector=detector, embedder=embedder, threshold=bench_threshold, mode="layer2"
    )
    df = df.with_columns(l2_bench["risk_hit"].alias("l2_bench_hit"))

    # 5. Chạy Combined (Benchmark threshold)
    comb_bench = compute_i8_combined(
        df, cfg=active_cfg, detector=detector, embedder=embedder, threshold=bench_threshold, mode="combined"
    )
    df = df.with_columns(comb_bench["risk_hit"].alias("comb_bench_hit"))

    results: dict[str, Any] = {
        "metadata": {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "input_file": str(input_path),
            "sample_size": df.height,
            "ground_truth_risk_count": int(df["expected_risk_hit"].sum()),
            "production_threshold": prod_threshold,
            "benchmark_threshold": bench_threshold,
            "embedding_model": active_cfg.default.nlp.embedding_model,
        },
        "methods": {
            "Layer 1 (Lexicon Only)": evaluate_method(df, "l1_hit"),
            f"Layer 2 Only (Prod th={prod_threshold:.2f})": evaluate_method(df, "l2_prod_hit"),
            f"Combined (Prod th={prod_threshold:.2f})": evaluate_method(df, "comb_prod_hit"),
            f"Layer 2 Only (Benchmark th={bench_threshold:.2f})": evaluate_method(df, "l2_bench_hit"),
            f"Combined (Benchmark th={bench_threshold:.2f})": evaluate_method(df, "comb_bench_hit"),
        },
    }

    # Xuất báo cáo ra file JSON
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        log.info("Đã lưu báo cáo đánh giá rủi ro tại: %s", output_path)

    # In kết quả trực quan ra console
    print("\n" + "=" * 80)
    print("📈 BẢNG SO SÁNH HIỆU NĂNG PHÁT HIỆN RỦI RO (I8) TRÊN SAMPLE REVIEWS")
    print("=" * 80)
    print(f"{'Phương pháp':<38} | {'Precision':<9} | {'Recall':<9} | {'F1':<9} | {'FP':<4} | {'FN':<4}")
    print("-" * 80)
    for name, m in results["methods"].items():
        print(
            f"{name:<38} | {m['precision']:<9.4f} | {m['recall']:<9.4f} | {m['f1']:<9.4f} | {m['fp']:<4} | {m['fn']:<4}"
        )
    print("=" * 80)

    # In phân tích ngắn các trường hợp FP/FN đáng chú ý
    comb_prod_res = results["methods"][f"Combined (Prod th={prod_threshold:.2f})"]
    if comb_prod_res["fp_samples"]:
        print(f"\n⚠️ Danh sách {len(comb_prod_res['fp_samples'])} ca False Positive (Combined prod th={prod_threshold:.2f}):")
        for s in comb_prod_res["fp_samples"]:
            print(f"  • [{s['review_id']}] sim={s['risk_sim']:.3f} | {s['text'][:70]}...")

    if comb_prod_res["fn_samples"]:
        print(f"\n⚠️ Danh sách {len(comb_prod_res['fn_samples'])} ca False Negative (Combined prod th={prod_threshold:.2f}):")
        for s in comb_prod_res["fn_samples"]:
            print(f"  • [{s['review_id']}] sim={s['risk_sim']:.3f} | {s['text'][:70]}...")
    print()

    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluation runner for BrandSentinel I8 Risk Detection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=REPO_ROOT / "tests" / "fixtures" / "sample_reviews.csv",
        help="Đường dẫn file sample_reviews.csv gán nhãn",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "tests" / "fixtures" / "risk_evaluation_report.json",
        help="Đường dẫn lưu báo cáo JSON kết quả",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.40,
        help="Ngưỡng tương đồng benchmark để so sánh đối ứng",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=None,
        help="Thư mục configs (mặc định configs/)",
    )

    args = parser.parse_args(argv)
    cfg = load_config(args.config_dir)

    run_evaluation(
        input_path=args.input,
        output_path=args.output,
        cfg=cfg,
        benchmark_threshold=args.threshold,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
