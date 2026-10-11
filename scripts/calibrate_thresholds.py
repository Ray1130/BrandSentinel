"""Calibrate score thresholds and evaluate synthetic crises for one category.

Example:
    uv run python scripts/calibrate_thresholds.py --category Baby_Products
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from brandsentinel.core.config import load_config
from brandsentinel.core.io import read_table
from brandsentinel.core.types import Table
from brandsentinel.evaluation.synthetic import calibrate_category

app = typer.Typer(no_args_is_help=True)


@app.command()
def main(
    category: Annotated[str, typer.Option(help="Category with saved feature_series")],
    config_dir: Annotated[
        Path | None,
        typer.Option(
            help="Directory containing default.yaml, indicators.yaml, and thresholds.yaml"
        ),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option(help="Write the JSON report to a file instead of stdout"),
    ] = None,
    max_false_alarm_rate: Annotated[
        float,
        typer.Option(min=0.0, max=1.0, help="Maximum false-alarm rate for trigger sweeps"),
    ] = 0.05,
    event_days: Annotated[
        int,
        typer.Option(min=1, help="Number of final windows used as the event"),
    ] = 14,
    no_tune_triggers: Annotated[
        bool,
        typer.Option("--no-tune-triggers", help="Skip the k/m/n and PELT penalty sweeps"),
    ] = False,
) -> None:
    """Create a calibration report without changing configuration files."""
    config = load_config(config_dir)
    feature_series = read_table(
        Table.FEATURE_SERIES,
        config,
        category=category,
        check=True,
    )
    report = calibrate_category(
        feature_series,
        config,
        max_false_alarm_rate=max_false_alarm_rate,
        event_days=event_days,
        tune_triggers=not no_tune_triggers,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    if output is None:
        typer.echo(rendered)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
        typer.echo(f"Calibration report: {output}")


if __name__ == "__main__":
    app()
