"""CLI: `uv run bs run --stage <stage> --category <name> --start <date> --end <date>`."""

from datetime import date
from enum import StrEnum
from typing import Annotated

import typer

from brandsentinel.pipeline import run_preprocess

app = typer.Typer(help="BrandSentinel CLI", no_args_is_help=True)


class Stage(StrEnum):
    preprocess = "preprocess"
    features = "features"
    detect = "detect"
    score = "score"
    explain = "explain"
    evaluate = "evaluate"


@app.callback()
def main() -> None:
    """BrandSentinel command line."""


@app.command()
def run(
    stage: Annotated[Stage, typer.Option("--stage", help="Stage cần chạy")],
    category: Annotated[str, typer.Option("--category", help="Category, ví dụ Baby_Products")],
    start: Annotated[date | None, typer.Option(help="Ngày bắt đầu YYYY-MM-DD (UTC)")] = None,
    end: Annotated[date | None, typer.Option(help="Ngày kết thúc YYYY-MM-DD (UTC)")] = None,
) -> None:
    """Chạy một stage của pipeline."""
    if stage is Stage.preprocess:
        clean, daily, audit_path = run_preprocess(category, start=start, end=end)
        typer.echo(
            f"Preprocessed {category}: {clean.height} reviews, {daily.height} SKU-days"
        )
        if audit_path is not None:
            typer.echo(f"Audit log: {audit_path}")
        return
    typer.echo(f"[stub] stage={stage.value} category={category} (chưa cài đặt)")
