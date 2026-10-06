"""CLI: `uv run bs run --stage <stage> --category <name> --start <date> --end <date>`."""

from enum import StrEnum
from typing import Annotated

import typer

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
    start: Annotated[str, typer.Option("--start", help="Ngày bắt đầu YYYY-MM-DD (UTC)")],
    end: Annotated[str, typer.Option("--end", help="Ngày kết thúc YYYY-MM-DD (UTC)")],
) -> None:
    """Chạy một stage của pipeline. TODO: nối với brandsentinel.pipeline.run."""
    typer.echo(f"[stub] stage={stage.value} category={category} {start}..{end} (chưa cài đặt)")
