from datetime import date

import polars as pl
import pytest

from brandsentinel.core.schemas import empty_table
from brandsentinel.core.types import Table
from brandsentinel.evaluation.ablation import (
    ablate_alerts,
    evaluate_signal_ablation,
    evaluate_tau_sensitivity,
    format_report_table,
    reclassify_with_tau,
)


def _mock_tables():
    # Alerts với 3 SKU mô phỏng các kịch bản kích hoạt chỉ báo khác nhau
    # Trọng số chuẩn (tổng 21 điểm):
    # Volume: I1 (1/21), I2 (1/21), I3 (2/21)
    # Rating: I4 (3/21), I5 (2/21), I6 (2/21)
    # Content: I7 (3/21), I8 (5/21), I9 (2/21)
    alerts = pl.DataFrame(
        [
            # SKU-1: Kích hoạt cả 3 nhóm (I1, I3, I4, I8) -> score = 11/21 ~ 0.52 (HIGH)
            (
                "SKU-1",
                "category",
                date(2025, 1, 1),
                0.52,
                "HIGH",
                ["volume", "rating", "content"],
                1,
                False,
                ["I1", "I3", "I4", "I8"],
            ),
            (
                "SKU-1",
                "category",
                date(2025, 1, 5),
                0.52,
                "HIGH",
                ["volume", "rating", "content"],
                2,
                False,
                ["I1", "I3", "I4", "I8"],
            ),
            # SKU-2: Kích hoạt volume + rating (I1, I4, I5) -> score = 6/21 ~ 0.29 (MEDIUM)
            (
                "SKU-2",
                "category",
                date(2025, 1, 3),
                0.29,
                "MEDIUM",
                ["volume", "rating"],
                1,
                False,
                ["I1", "I4", "I5"],
            ),
            # SKU-3: Kích hoạt rating + content (I4, I8) -> score = 8/21 ~ 0.38 (MEDIUM)
            (
                "SKU-3",
                "category",
                date(2025, 1, 4),
                0.38,
                "MEDIUM",
                ["rating", "content"],
                1,
                False,
                ["I4", "I8"],
            ),
        ],
        schema=empty_table(Table.ALERTS).schema,
        orient="row",
    )

    labels = pl.DataFrame(
        [
            (
                "SKU-1",
                "category",
                "R-1",
                date(2025, 1, 10),
                "CPSC",
                "https://example.com/recall-1",
                "Product 1",
                "exact_asin",
                1.0,
            ),
            (
                "SKU-2",
                "category",
                "R-2",
                date(2025, 1, 10),
                "CPSC",
                "https://example.com/recall-2",
                "Product 2",
                "exact_asin",
                1.0,
            ),
            (
                "SKU-3",
                "category",
                "R-3",
                date(2025, 1, 10),
                "CPSC",
                "https://example.com/recall-3",
                "Product 3",
                "exact_asin",
                1.0,
            ),
        ],
        schema=empty_table(Table.RECALL_LABELS).schema,
        orient="row",
    )
    return alerts, labels


def test_ablate_alerts_drops_indicators_and_recalculates_score():
    alerts, _ = _mock_tables()

    # Bỏ nhóm content (I7, I8, I9)
    ablated = ablate_alerts(alerts, drop_group="content")

    # SKU-3 ban đầu có I4 (rating) và I8 (content)
    # Bỏ content, SKU-3 chỉ còn I4 (score = 3/21 = 0.14 < tau1=0.20) -> bị loại khỏi alerts
    assert "SKU-3" not in ablated["sku"].to_list()

    # SKU-1 có I1, I3 (volume) và I4 (rating) -> không còn I8 trong triggered_ids
    sku1_rows = ablated.filter(pl.col("sku") == "SKU-1")
    assert sku1_rows.height > 0
    for row in sku1_rows.iter_rows(named=True):
        assert "content" not in row["groups_active"]
        assert "I8" not in row["triggered_ids"]
        assert "I1" in row["triggered_ids"]
        assert "I3" in row["triggered_ids"]
        assert "I4" in row["triggered_ids"]
        # Do mất nhóm content (chỉ còn 2 nhóm active) và không có I8 severe override,
        # SKU-1 không thể giữ mức HIGH, phải hạ xuống MEDIUM
        assert row["level"] == "MEDIUM"


def test_ablate_alerts_keep_only_group():
    alerts, _ = _mock_tables()

    # Chỉ giữ nhóm content
    ablated = ablate_alerts(alerts, keep_only_group="content")

    # SKU-2 không có content -> bị loại
    assert "SKU-2" not in ablated["sku"].to_list()

    # Các dòng còn lại chỉ có nhóm content
    for row in ablated.iter_rows(named=True):
        assert row["groups_active"] == ["content"]
        for iid in row["triggered_ids"]:
            assert iid in ("I7", "I8", "I9")


def test_ablate_alerts_validations():
    alerts, _ = _mock_tables()

    with pytest.raises(ValueError, match="Chỉ được chỉ định một trong hai"):
        ablate_alerts(alerts, drop_group="volume", keep_only_group="rating")

    empty = empty_table(Table.ALERTS)
    res = ablate_alerts(empty, drop_group="volume")
    assert res.is_empty()
    assert res.schema == empty.schema


def test_reclassify_with_tau():
    alerts, _ = _mock_tables()

    # Ngưỡng rất cao: tau1=0.60, tau2=0.80 -> các alert có score < 0.60 bị loại hết
    strict = reclassify_with_tau(alerts, tau1=0.60, tau2=0.80)
    assert strict.is_empty()

    # Ngưỡng lỏng: tau1=0.10, tau2=0.30 -> toàn bộ 4 dòng đều thỏa mãn
    loose = reclassify_with_tau(alerts, tau1=0.10, tau2=0.30)
    assert loose.height == alerts.height

    # Tại ngưỡng lỏng tau2=0.30, SKU-1 có 3 nhóm và score 0.52 > 0.30 -> chắc chắn HIGH
    assert loose.filter(pl.col("sku") == "SKU-1")["level"].to_list() == ["HIGH", "HIGH"]

    # Ngưỡng không hợp lệ: tau1 >= tau2
    with pytest.raises(ValueError, match="Ngưỡng không hợp lệ"):
        reclassify_with_tau(alerts, tau1=0.50, tau2=0.30)


def test_evaluate_signal_ablation():
    alerts, labels = _mock_tables()

    report_df = evaluate_signal_ablation(
        alerts, labels, lookback_days=10, include_single_group=True
    )

    configs = report_df["configuration"].to_list()
    assert "Full Model" in configs
    assert "w/o volume" in configs
    assert "w/o rating" in configs
    assert "w/o content" in configs
    assert "only volume" in configs
    assert "only rating" in configs
    assert "only content" in configs

    full_row = report_df.filter(pl.col("configuration") == "Full Model").to_dicts()[0]
    assert full_row["recall_events"] == 3
    assert full_row["delta_recall_pct"] == 0.0

    # Khi bỏ content, SKU-3 mất cảnh báo -> detected_events giảm hoặc bằng Full Model
    wo_content_row = report_df.filter(pl.col("configuration") == "w/o content").to_dicts()[0]
    assert wo_content_row["detected_events"] <= full_row["detected_events"]


def test_evaluate_tau_sensitivity():
    alerts, labels = _mock_tables()

    tau_pairs = [(0.15, 0.30), (0.20, 0.40), (0.50, 0.70)]
    sens_df = evaluate_tau_sensitivity(alerts, labels, tau_pairs=tau_pairs, lookback_days=10)

    assert sens_df.height == 3
    # Khi ngưỡng tăng, tổng số alert phát ra phải giảm dần hoặc giữ nguyên
    total_alerts = sens_df["total_alerts"].to_list()
    assert total_alerts[0] >= total_alerts[1] >= total_alerts[2]


def test_format_report_table():
    alerts, labels = _mock_tables()
    df = evaluate_signal_ablation(alerts, labels, lookback_days=10)

    md = format_report_table(df, title="Signal Group Ablation Study", format_type="markdown")
    assert "### Signal Group Ablation Study" in md
    assert "| configuration |" in md
    assert "| Full Model |" in md

    latex = format_report_table(df, title="Signal Group Ablation Study", format_type="latex")
    assert "\\begin{tabular}" in latex
    assert "\\end{tabular}" in latex

    empty_df = pl.DataFrame()
    empty_md = format_report_table(empty_df, title="Empty Table")
    assert "(Không có dữ liệu)" in empty_md

    with pytest.raises(ValueError, match="Định dạng không hỗ trợ"):
        format_report_table(df, format_type="html")
