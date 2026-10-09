from datetime import date, timedelta

import polars as pl
from typer.testing import CliRunner

from brandsentinel import cli
from brandsentinel.core.io import read_table, write_table
from brandsentinel.core.types import Level, Table
from brandsentinel.pipeline import run_score
from brandsentinel.scoring.score import score_indicators
from brandsentinel.scoring.state import apply_hysteresis
from brandsentinel.scoring.thresholds import calibrate_thresholds


def make_matrix(windows: list[set[str]]) -> pl.DataFrame:
    rows = []
    start = date(2025, 1, 1)
    for offset, active_ids in enumerate(windows):
        for indicator_number in range(1, 12):
            indicator_id = f"I{indicator_number}"
            rows.append(
                {
                    "sku": "SKU-1",
                    "category": "Test_Category",
                    "window_end": start + timedelta(days=offset),
                    "indicator_id": indicator_id,
                    "triggered": indicator_id in active_ids,
                    "strength": 3.0 if indicator_id in active_ids else 0.0,
                }
            )
    return pl.DataFrame(rows)


def test_score_indicators_weights_groups_persistence_and_verification(cfg):
    matrix = make_matrix(
        [
            {"I4", "I5", "I10"},
            {"I4", "I5"},
            set(),
            set(),
            set(),
        ]
    )

    alerts = score_indicators(matrix, cfg)

    assert alerts["score"].to_list() == [5 / 21, 5 / 21, 0.0, 0.0, 0.0]
    assert alerts["groups_active"].to_list() == [["rating"], ["rating"], [], [], []]
    assert alerts["persist"].to_list() == [1, 2, 0, 0, 0]
    assert alerts["verify_flag"].to_list() == [True, False, False, False, False]
    assert alerts["level"].to_list() == ["LOW", "MEDIUM", "MEDIUM", "MEDIUM", "LOW"]
    assert alerts["triggered_ids"].to_list()[0] == ["I4", "I5", "I10"]


def test_high_requires_cross_group_confirmation_or_severe_override(cfg):
    matrix = make_matrix(
        [
            {"I1", "I4", "I5", "I7"},
            {"I4", "I5", "I8"},
        ]
    )

    alerts = score_indicators(matrix, cfg)

    assert alerts["level"].to_list() == ["HIGH", "HIGH"]
    assert alerts["groups_active"].to_list() == [
        ["content", "rating", "volume"],
        ["content", "rating"],
    ]


def test_i8_severe_override_requires_severe_strength(cfg):
    matrix = make_matrix(
        [
            {"I4", "I5", "I6", "I8"},
            {"I4", "I5", "I6", "I8"},
        ]
    ).with_columns(
        pl.when(pl.col("indicator_id") == "I8")
        .then(pl.when(pl.col("window_end") == date(2025, 1, 1)).then(0.5).otherwise(1.0))
        .otherwise(pl.col("strength"))
        .alias("strength")
    )

    alerts = score_indicators(matrix, cfg)

    assert alerts["level"].to_list() == ["MEDIUM", "HIGH"]


def test_verification_flag_obeys_configured_all_requirement(cfg):
    flags = cfg.thresholds.flags.model_copy(update={"require": "all"})
    thresholds = cfg.thresholds.model_copy(update={"flags": flags})
    all_flags_config = cfg.model_copy(update={"thresholds": thresholds})

    alerts = score_indicators(make_matrix([{"I10"}]), all_flags_config)

    assert alerts["verify_flag"].to_list() == [False]


def test_threshold_calibration_uses_percentiles_and_floors(cfg):
    calibrated = calibrate_thresholds([0.1] * 200, cfg)

    assert calibrated.tau1 == cfg.thresholds.calibration.floors.tau1_min
    assert calibrated.tau2 == cfg.thresholds.calibration.floors.tau2_min


def test_hysteresis_delays_downgrades_and_escalates_immediately(cfg):
    levels = apply_hysteresis(
        [Level.HIGH, Level.LOW, Level.LOW, Level.LOW, Level.HIGH],
        cfg,
    )

    assert levels == [Level.HIGH, Level.HIGH, Level.HIGH, Level.LOW, Level.HIGH]


def test_run_score_persists_alerts_idempotently(cfg, tmp_path):
    isolated_config = cfg.model_copy(update={"root": tmp_path})
    matrix = make_matrix([{"I1", "I4", "I7"}])
    write_table(matrix, Table.INDICATOR_MATRIX, isolated_config)

    first = run_score("Test_Category", config=isolated_config)
    second = run_score("Test_Category", config=isolated_config)
    stored = read_table(Table.ALERTS, isolated_config, category="Test_Category")

    assert first.equals(second)
    assert stored.equals(first)


def test_cli_score_runs_and_reports_alerts(monkeypatch, cfg, tmp_path):
    isolated_config = cfg.model_copy(update={"root": tmp_path})
    write_table(
        make_matrix([{"I1", "I4", "I5", "I7"}]),
        Table.INDICATOR_MATRIX,
        isolated_config,
    )
    monkeypatch.setattr(cli, "load_config", lambda: isolated_config)

    result = CliRunner().invoke(
        cli.app,
        ["run", "--stage", "score", "--category", "Test_Category"],
    )

    assert result.exit_code == 0, result.output
    assert "alerts: 1" in result.output
    assert "'HIGH': 1" in result.output
