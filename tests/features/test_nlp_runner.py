"""Unit and integration tests for production runner and evaluation scripts (Phase 3B-1)."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import polars as pl
import pytest

from brandsentinel.core.io import write_table
from brandsentinel.core.types import Table
from brandsentinel.testing.mock_data import make_clean_reviews
from scripts.evaluate_risk import evaluate_method, run_evaluation
from scripts.run_nlp import (
    compute_data_quality_report,
    resolve_categories,
    run_nlp_for_category,
)


def test_category_resolution(cfg):
    """1. Test that category aliases map to configured categories."""
    assert resolve_categories("dev", cfg) == ["Baby_Products"]
    assert resolve_categories("holdout", cfg) == ["Home_and_Kitchen"]
    assert resolve_categories("CustomCategory", cfg) == ["CustomCategory"]


def test_runner_missing_clean_reviews_exits_cleanly(cfg):
    """2. Test runner behavior when clean_reviews table does not exist."""
    nlp_df, written, report = run_nlp_for_category(
        category="NonExistentCategory_XYZ",
        cfg=cfg,
        write=False,
    )
    # Must return None cleanly without unhandled exception
    assert nlp_df is None
    assert written == []
    assert report is None


def test_runner_execution_and_write_path_on_mock(tmp_path, cfg):
    """3. Test successful execution, quality reporting, and parquet writing on mock."""
    mock_clean = make_clean_reviews(seed=42).head(20)

    # Ghi mock clean_reviews vào interim zone tạm thời
    written_clean = write_table(mock_clean, Table.CLEAN_REVIEWS, cfg, check=True)
    assert len(written_clean) > 0

    cat = mock_clean["category"][0]
    nlp_df, written_paths, report = run_nlp_for_category(
        category=cat,
        cfg=cfg,
        write=True,
    )

    assert nlp_df is not None
    assert nlp_df.height == 20
    assert len(written_paths) > 0
    for p in written_paths:
        assert p.exists()
        assert p.suffix == ".parquet"
        assert f"category={cat}" in str(p)

    # Báo cáo chất lượng dữ liệu
    assert report is not None
    assert report["input"]["rows"] == 20
    assert report["output"]["rows"] == 20
    assert report["output"]["throughput_reviews_per_sec"] > 0


def test_data_quality_report_empty_and_null_metrics():
    """4. Test compute_data_quality_report calculations."""
    clean_empty = pl.DataFrame(schema={"review_id": pl.String, "text_norm": pl.String})
    nlp_empty = pl.DataFrame(
        schema={
            "review_id": pl.String,
            "risk_hit": pl.Boolean,
            "mismatch": pl.Boolean,
            "aspect": pl.String,
        }
    )
    report = compute_data_quality_report(clean_empty, nlp_empty, runtime_seconds=0.0)
    assert report["input"]["rows"] == 0
    assert report["output"]["rows"] == 0
    assert report["output"]["throughput_reviews_per_sec"] == 0.0


def test_evaluation_runner_correctness(tmp_path, cfg):
    """5. Test evaluate_risk.py execution and output structure."""
    sample_csv = Path("tests/fixtures/sample_reviews.csv")
    output_json = tmp_path / "eval_report.json"

    results = run_evaluation(
        input_path=sample_csv,
        output_path=output_json,
        cfg=cfg,
        benchmark_threshold=0.40,
    )

    assert output_json.exists()
    assert "metadata" in results
    assert "methods" in results
    assert results["metadata"]["sample_size"] == 70

    l1 = results["methods"]["Layer 1 (Lexicon Only)"]
    assert l1["precision"] == 1.0
    assert l1["recall"] == 1.0
    assert l1["f1"] == 1.0
    assert l1["confusion_matrix"]["tp"] == 14
    assert l1["confusion_matrix"]["tn"] == 56

    comb_prod = results["methods"]["Combined (Prod th=0.60)"]
    assert comb_prod["recall"] == 1.0
    assert comb_prod["precision"] > 0.90
    assert comb_prod["confusion_matrix"]["tp"] == 14


def test_evaluate_method_helper():
    """6. Test evaluate_method with mock binary predictions."""
    df = pl.DataFrame(
        {
            "review_id": ["R1", "R2", "R3", "R4"],
            "expected_risk_hit": [True, True, False, False],
            "pred": [True, False, True, False],
            "text_norm": ["t1", "t2", "t3", "t4"],
            "risk_sim": [0.8, 0.2, 0.7, 0.1],
            "risk_terms": [["fire"], None, ["fake"], None],
            "expected_risk_terms": [["fire"], ["shock"], None, None],
        }
    )
    metrics = evaluate_method(df, pred_col="pred", true_col="expected_risk_hit")
    assert metrics["tp"] == 1
    assert metrics["fn"] == 1
    assert metrics["fp"] == 1
    assert metrics["tn"] == 1
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == 0.5
    assert len(metrics["fp_samples"]) == 1
    assert len(metrics["fn_samples"]) == 1
    assert metrics["fp_samples"][0]["review_id"] == "R3"
    assert metrics["fn_samples"][0]["review_id"] == "R2"
