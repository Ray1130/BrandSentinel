"""Run baseline comparison, signal group ablation, and threshold sensitivity evaluation.

Usage examples:
    uv run python scripts/run_eval.py --mock
    uv run python scripts/run_eval.py --category Baby_Products
    uv run python scripts/run_eval.py --mock --output reports/evaluation_report.md
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import polars as pl
import typer

from brandsentinel.core.config import Config, load_config
from brandsentinel.core.io import read_table
from brandsentinel.core.types import Table
from brandsentinel.evaluation.ablation import (
    evaluate_signal_ablation,
    evaluate_tau_sensitivity,
    format_report_table,
)
from brandsentinel.evaluation.baselines import evaluate_baselines
from brandsentinel.testing.mock_data import make_all

app = typer.Typer(help="BrandSentinel Evaluation & Ablation Runner", no_args_is_help=True)


def _format_baselines_table(results: dict[str, dict[str, object]]) -> str:
    rows = []
    for policy, metrics in results.items():
        rec = metrics.get("event_recall")
        mean_lt = metrics.get("mean_lead_time_days")
        rows.append(
            {
                "policy": policy,
                "recall_events": metrics.get("recall_events"),
                "detected_events": metrics.get("detected_events"),
                "missed_events": metrics.get("missed_events"),
                "event_recall_pct": round(rec * 100, 2) if isinstance(rec, float) else None,
                "mean_lead_time_days": round(mean_lt, 1) if isinstance(mean_lt, float) else None,
                "alerts_in_window": metrics.get("alerts_in_window"),
            }
        )
    df = pl.DataFrame(rows)
    return format_report_table(df, title="Policy Baselines Comparison", format_type="markdown")


@app.command()
def main(
    category: Annotated[
        str | None,
        typer.Option(help="Category name with saved alerts and recall_labels"),
    ] = None,
    mock: Annotated[
        bool,
        typer.Option("--mock", help="Use synthetic mock data instead of reading from disk"),
    ] = False,
    lookback_days: Annotated[
        int,
        typer.Option(min=1, help="Lookback window in days before recall date"),
    ] = 90,
    config_dir: Annotated[
        Path | None,
        typer.Option(help="Directory containing config files"),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option(help="Write the combined markdown report to a file instead of stdout"),
    ] = None,
) -> None:
    """Run baseline, signal group ablation, and tau sensitivity evaluations."""
    if not mock and not category:
        typer.echo(
            "Lỗi: Cần cung cấp --category hoặc thêm cờ --mock để chạy dữ liệu giả lập.",
            err=True,
        )
        raise typer.Exit(code=1)

    cfg: Config = load_config(config_dir)

    if mock:
        typer.echo("[INFO] Đang khởi tạo dữ liệu mock (Table.ALERTS, Table.RECALL_LABELS)...")
        tables = make_all(seed=42)
        alerts = tables[Table.ALERTS]
        recall_labels = tables[Table.RECALL_LABELS]
    else:
        typer.echo(f"[INFO] Đang đọc dữ liệu cho category={category}...")
        alerts = read_table(Table.ALERTS, cfg, category=category, check=True)
        recall_labels = read_table(Table.RECALL_LABELS, cfg, category=category, check=True)

    typer.echo(
        f"[INFO] Đánh giá trên {alerts.height} alerts và {recall_labels.height} sự kiện thu hồi..."
    )

    # 1. Baseline Evaluation
    baseline_res = evaluate_baselines(alerts, recall_labels, lookback_days=lookback_days)
    baseline_table = _format_baselines_table(baseline_res)

    # 2. Signal Group Ablation
    ablation_df = evaluate_signal_ablation(
        alerts,
        recall_labels,
        config=cfg,
        lookback_days=lookback_days,
        include_single_group=True,
    )
    ablation_table = format_report_table(
        ablation_df,
        title="Signal Group Ablation Study (Loại Bỏ Từng Nhóm Tín Hiệu)",
        format_type="markdown",
    )

    # 3. Tau Sensitivity Analysis
    sensitivity_df = evaluate_tau_sensitivity(
        alerts,
        recall_labels,
        config=cfg,
        lookback_days=lookback_days,
    )
    sensitivity_table = format_report_table(
        sensitivity_df,
        title="Threshold Sensitivity Analysis (Độ Nhạy Theo Ngưỡng τ)",
        format_type="markdown",
    )

    # Tổng hợp báo cáo Markdown
    full_report = "\n\n".join(
        [
            "# BrandSentinel Evaluation & Calibration Report",
            f"- **Category:** {category or 'Mock Synthetic Data'}",
            f"- **Lookback Days:** {lookback_days} days",
            f"- **Total Alerts:** {alerts.height}",
            f"- **Verified Recall Events:** {recall_labels.height}",
            "---",
            baseline_table,
            "---",
            ablation_table,
            "---",
            sensitivity_table,
        ]
    )

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(full_report + "\n", encoding="utf-8")
        typer.echo(f"[DONE] Đã lưu báo cáo thành công tại: {output}")
    else:
        typer.echo("\n" + full_report)


if __name__ == "__main__":
    app()
