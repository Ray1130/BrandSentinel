"""CLI: `uv run bs run --stage <stage> --category <name> --start <date> --end <date>`."""

from datetime import date
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer

from brandsentinel import pipeline
from brandsentinel.core.config import load_config

app = typer.Typer(help="BrandSentinel CLI", no_args_is_help=True)


class Stage(StrEnum):
    preprocess = "preprocess"
    features = "features"
    detect = "detect"
    score = "score"
    explain = "explain"
    evaluate = "evaluate"


IMPLEMENTED = {"preprocess", "features"}

NLP_MODULES = {
    "torch",
    "transformers",
    "bertopic",
    "sentence_transformers",
}


def _parse_date(value: str | None, option: str) -> date | None:
    if value is None:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError as e:
        raise typer.BadParameter(f"{option} must use YYYY-MM-DD") from e


@app.callback()
def main() -> None:
    """BrandSentinel command line."""


@app.command()
def run(
    stage: Annotated[
        Stage,
        typer.Option("--stage", help="Stage cần chạy"),
    ],
    category: Annotated[
        str,
        typer.Option("--category", help="Category, ví dụ Baby_Products"),
    ],
    start: Annotated[
        str | None,
        typer.Option(help="Ngày bắt đầu YYYY-MM-DD (UTC)"),
    ] = None,
    end: Annotated[
        str | None,
        typer.Option(help="Ngày kết thúc YYYY-MM-DD (UTC)"),
    ] = None,
    source: Annotated[
        Path | None,
        typer.Option(help="File raw cho preprocess (mặc định data/raw/<category>.jsonl)"),
    ] = None,
    no_nlp: Annotated[
        bool,
        typer.Option(
            "--no-nlp",
            help="features: bỏ NLP, chỉ I1-I6 (không cần extra)",
        ),
    ] = False,
) -> None:
    """Chạy một stage của pipeline."""

    if stage.value not in IMPLEMENTED:
        typer.echo(f"[stub] stage={stage.value} category={category} (chưa cài đặt)")
        return

    try:
        summary = pipeline.run_stage(
            stage.value,
            category,
            start=_parse_date(start, "--start"),
            end=_parse_date(end, "--end"),
            source=source,
            with_nlp=not no_nlp,
            config=load_config(),
        )

    except FileNotFoundError as e:
        typer.echo(f"Lỗi: {e}", err=True)
        raise typer.Exit(code=1) from e

    except ModuleNotFoundError as e:
        if e.name and e.name.split(".")[0] in NLP_MODULES:
            typer.echo(
                f"Thiếu thư viện NLP '{e.name}': chạy `uv sync --extra nlp`, "
                "hoặc thêm --no-nlp để chỉ tính I1-I6.",
                err=True,
            )
            raise typer.Exit(code=1) from e
        raise

    for key, value in summary.items():
        typer.echo(f"{key}: {value}")
