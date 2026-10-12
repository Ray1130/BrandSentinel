"""Integration and idempotency checks for `bs run` pipeline stages."""

from datetime import UTC, date, datetime, timedelta

import numpy as np
import polars as pl
import pytest
from typer.testing import CliRunner

from brandsentinel import cli, pipeline
from brandsentinel.core.config import load_config
from brandsentinel.core.io import read_table, write_table
from brandsentinel.core.types import Table
from brandsentinel.testing.mock_data import make_all, make_clean_reviews, make_nlp_features

KEYS = {
    Table.CLEAN_REVIEWS: ["review_id"],
    Table.DAILY_AGG: ["sku", "date"],
    Table.NLP_FEATURES: ["review_id"],
    Table.FEATURE_SERIES: ["sku", "window_end", "indicator_id", "feature"],
    Table.INDICATOR_MATRIX: ["sku", "window_end", "indicator_id"],
    Table.ALERTS: ["sku", "window_end"],
}


@pytest.fixture
def cfg(tmp_path):
    base = load_config()
    return base.model_copy(update={"root": tmp_path})


def snapshot(cfg, table, category):
    df = read_table(table, cfg, category=category)
    return df.sort(KEYS[table])


def assert_same_no_dups(first, second, table):
    assert first.height > 0
    assert second.height == first.height, f"{table}: {first.height} -> {second.height} dong"
    assert not second.select(KEYS[table]).is_duplicated().any(), f"{table}: trung khoa"
    assert first.equals(second), f"{table}: noi dung doi sau khi chay lai"


# ---------------------------------------------------------------- features
@pytest.fixture
def stored(cfg):
    tables = make_all(seed=42, cfg=cfg)
    category = tables[Table.CLEAN_REVIEWS]["category"][0]
    write_table(tables[Table.CLEAN_REVIEWS], Table.CLEAN_REVIEWS, cfg)
    write_table(tables[Table.DAILY_AGG], Table.DAILY_AGG, cfg)
    return category, tables


def make_runner(tables, calls):
    def runner(clean, _cfg):
        calls.append(clean.height)
        nlp = tables[Table.NLP_FEATURES]
        return nlp.filter(pl.col("review_id").is_in(clean["review_id"]))

    return runner


def test_features_rerun_no_duplicates(cfg, stored):
    pytest.importorskip("torch", reason="requires the optional nlp extra")
    category, tables = stored
    calls: list[int] = []
    runner = make_runner(tables, calls)
    watched = (Table.FEATURE_SERIES, Table.NLP_FEATURES)

    pipeline.run_features(category, config=cfg, nlp_runner=runner)
    first = {t: snapshot(cfg, t, category) for t in watched}
    pipeline.run_features(category, config=cfg, nlp_runner=runner)
    second = {t: snapshot(cfg, t, category) for t in watched}

    for t in watched:
        assert_same_no_dups(first[t], second[t], t)
    assert len(calls) == 1, "lan chay thu hai khong duoc tinh lai NLP"


def test_features_partial_range_rerun_keeps_everything(cfg, stored):
    pytest.importorskip("torch", reason="requires the optional nlp extra")
    category, tables = stored
    runner = make_runner(tables, [])
    pipeline.run_features(category, config=cfg, nlp_runner=runner)
    full = snapshot(cfg, Table.FEATURE_SERIES, category)

    lo, hi = full["window_end"].min(), full["window_end"].max()
    pipeline.run_features(
        category,
        config=cfg,
        nlp_runner=runner,
        start=lo + timedelta(days=30),
        end=hi - timedelta(days=30),
    )
    assert_same_no_dups(full, snapshot(cfg, Table.FEATURE_SERIES, category), Table.FEATURE_SERIES)


def test_no_nlp_refuses_to_drop_content_features(cfg, stored):
    category, tables = stored
    write_table(tables[Table.FEATURE_SERIES], Table.FEATURE_SERIES, cfg)
    before = snapshot(cfg, Table.FEATURE_SERIES, category)
    with pytest.raises(RuntimeError, match="run_features"):
        pipeline.run_features(category, config=cfg, with_nlp=False)
    assert before.equals(snapshot(cfg, Table.FEATURE_SERIES, category))


def test_cli_features_twice(monkeypatch, cfg, stored):
    category, _ = stored
    monkeypatch.setattr(cli, "load_config", lambda: cfg)

    args = ["run", "--stage", "features", "--category", category, "--no-nlp"]
    result = CliRunner().invoke(cli.app, args)
    assert result.exit_code == 0, result.output
    first = snapshot(cfg, Table.FEATURE_SERIES, category)
    result = CliRunner().invoke(cli.app, args)
    assert result.exit_code == 0, result.output
    assert_same_no_dups(first, snapshot(cfg, Table.FEATURE_SERIES, category), Table.FEATURE_SERIES)


def test_cli_features_without_preprocess_fails_cleanly(monkeypatch, cfg):
    monkeypatch.setattr(cli, "load_config", lambda: cfg)
    result = CliRunner().invoke(cli.app, ["run", "--stage", "features", "--category", "Nope"])
    assert result.exit_code == 1
    assert "preprocess" in result.output


# -------------------------------------------------------------- preprocess
STARS = [1, 2, 3, 4, 5]
STAR_P = [0.07, 0.05, 0.08, 0.2, 0.6]


def _raw_reviews(n_sku=4, days=240, per_day=6, seed=0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    t0 = datetime(2025, 1, 1, tzinfo=UTC)
    rows = []
    for s in range(n_sku):
        for d in range(days):
            for k in range(per_day):
                ts = t0 + timedelta(days=d, hours=int(rng.integers(0, 24)), minutes=k)
                rows.append(
                    {
                        "parent_asin": f"B0RAW{s:05d}",
                        "rating": float(rng.choice(STARS, p=STAR_P)),
                        "title": f"title {s}-{d}-{k}",
                        "text": f"review text {s}-{d}-{k} quality {int(rng.integers(0, 1000))}",
                        "user_id": f"U{int(rng.integers(0, 10**9))}",
                        "timestamp": int(ts.timestamp() * 1000),
                        "verified_purchase": bool(rng.random() < 0.8),
                    }
                )
    df = pl.DataFrame(rows)
    duplicate_keys = df.head(5).with_columns(pl.col("timestamp") + 1)
    return pl.concat([df, duplicate_keys])  # Same dedup key, distinct review IDs.


def test_preprocess_rerun_no_duplicates(cfg):
    category, raw = "Test_Category", _raw_reviews()
    tables = (Table.CLEAN_REVIEWS, Table.DAILY_AGG)

    pipeline.run_preprocess(category, source=raw, config=cfg)
    first = {t: snapshot(cfg, t, category) for t in tables}
    assert first[Table.DAILY_AGG].height > 0, "khong SKU nao du dieu kien: xem data.sku_selection"
    pipeline.run_preprocess(category, source=raw, config=cfg)
    second = {t: snapshot(cfg, t, category) for t in tables}

    for t in tables:
        assert_same_no_dups(first[t], second[t], t)


def test_cli_pipeline_two_categories_rerun_no_duplicates(monkeypatch, cfg, tmp_path):
    categories = ("Baby_Products", "Home_and_Kitchen")
    source_paths = {}
    for category in categories:
        clean = make_clean_reviews(
            n_skus=3,
            days=190,
            seed=42,
            category=category,
            start=date(2023, 1, 2),
        )
        source = clean.select(
            pl.col("sku").alias("parent_asin"),
            "rating",
            pl.col("text_raw").alias("text"),
            "user_id",
            pl.col("ts").dt.epoch("ms").alias("timestamp"),
            "verified_purchase",
        )
        source_path = tmp_path / f"{category}.jsonl"
        source.write_ndjson(source_path)
        source_paths[category] = source_path

    nlp_calls = []

    def mock_nlp(clean, _cfg):
        nlp_calls.append(clean.height)
        return make_nlp_features(clean, seed=42)

    monkeypatch.setattr(cli, "load_config", lambda: cfg)
    monkeypatch.setattr(pipeline, "_default_nlp_runner", mock_nlp)

    stage_outputs = {
        "preprocess": (Table.CLEAN_REVIEWS, Table.DAILY_AGG),
        "features": (Table.NLP_FEATURES, Table.FEATURE_SERIES),
        "detect": (Table.INDICATOR_MATRIX,),
        "score": (Table.ALERTS,),
    }

    def run_pipeline():
        for category in categories:
            for stage in stage_outputs:
                args = ["run", "--stage", stage, "--category", category]
                if stage == "preprocess":
                    args.extend(["--source", str(source_paths[category])])
                result = CliRunner().invoke(cli.app, args)
                assert result.exit_code == 0, f"{stage}/{category}: {result.output}"

    run_pipeline()
    first = {
        (category, table): snapshot(cfg, table, category)
        for category in categories
        for tables in stage_outputs.values()
        for table in tables
    }
    for category in categories:
        matrix = first[(category, Table.INDICATOR_MATRIX)]
        assert set(matrix["indicator_id"].unique()) == {f"I{i}" for i in range(1, 12)}

    run_pipeline()
    second = {
        (category, table): snapshot(cfg, table, category)
        for category in categories
        for tables in stage_outputs.values()
        for table in tables
    }
    for key in first:
        table = key[1]
        assert_same_no_dups(first[key], second[key], table)
    assert len(nlp_calls) == len(categories), "NLP chỉ nên tính ở lần đầu cho mỗi category"
