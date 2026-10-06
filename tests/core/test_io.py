import json
import shutil
from datetime import date
from pathlib import Path

import polars as pl
import pytest

from brandsentinel.core.config import load_config
from brandsentinel.core.io import (
    partition_path,
    read_json,
    read_table,
    recommendation_path,
    write_json,
    write_table,
)
from brandsentinel.core.logging import AuditLog
from brandsentinel.core.types import Table

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def cfg_tmp(tmp_path):
    shutil.copytree(REPO / "configs", tmp_path / "configs")
    return load_config(tmp_path / "configs")  # root = tmp_path -> data/ nằm trong tmp


def test_roundtrip(cfg_tmp, mock_tables):
    for table in (Table.CLEAN_REVIEWS, Table.DAILY_AGG, Table.ALERTS):
        df = mock_tables[table]
        write_table(df, table, cfg_tmp)
        back = read_table(table, cfg_tmp, check=True)
        assert back.height == df.height
        assert back.schema == df.schema


def test_write_is_idempotent(cfg_tmp, mock_tables):
    df = mock_tables[Table.DAILY_AGG]
    write_table(df, Table.DAILY_AGG, cfg_tmp)
    write_table(df, Table.DAILY_AGG, cfg_tmp)
    assert read_table(Table.DAILY_AGG, cfg_tmp).height == df.height


def test_rewrite_replaces_only_same_sku_and_range(cfg_tmp, mock_tables):
    df = mock_tables[Table.DAILY_AGG]
    write_table(df, Table.DAILY_AGG, cfg_tmp)
    sku = "B0MOCK0002"
    fix = df.filter((pl.col("sku") == sku) & (pl.col("date") <= date(2023, 2, 15))).with_columns(
        pl.lit(0).cast(pl.Int32).alias("n_verified_all"),
        pl.lit(0).cast(pl.Int32).alias("n_verified"),
    )
    write_table(fix, Table.DAILY_AGG, cfg_tmp)
    back = read_table(Table.DAILY_AGG, cfg_tmp)
    assert back.height == df.height
    mine = back.filter((pl.col("sku") == sku) & (pl.col("date") <= date(2023, 2, 15)))
    assert mine["n_verified_all"].sum() == 0
    other = back.filter(pl.col("sku") != sku).sort(["sku", "date"])
    assert other.equals(df.filter(pl.col("sku") != sku).sort(["sku", "date"]))


def test_read_filters(cfg_tmp, mock_tables):
    write_table(mock_tables[Table.DAILY_AGG], Table.DAILY_AGG, cfg_tmp)
    sub = read_table(
        Table.DAILY_AGG, cfg_tmp, start=date(2023, 2, 1), end=date(2023, 2, 28), skus=["B0MOCK0000"]
    )
    assert sub["sku"].unique().to_list() == ["B0MOCK0000"]
    assert sub["date"].min() >= date(2023, 2, 1) and sub["date"].max() <= date(2023, 2, 28)


def test_missing_data_and_unsafe_category(cfg_tmp):
    with pytest.raises(FileNotFoundError):
        read_table(Table.ALERTS, cfg_tmp)
    with pytest.raises(ValueError):
        partition_path(cfg_tmp, Table.ALERTS, "Home:Kitchen", "2023-01")  # ký tự cấm trên Windows


def test_invalid_data_not_written(cfg_tmp, mock_tables):
    bad = mock_tables[Table.DAILY_AGG].with_columns(pl.lit(-1).cast(pl.Int32).alias("n"))
    with pytest.raises(Exception):  # noqa: B017
        write_table(bad, Table.DAILY_AGG, cfg_tmp)
    assert not (cfg_tmp.path("data_interim") / "daily_agg").exists()


def test_json_utf8_and_audit(cfg_tmp):
    path = recommendation_path(cfg_tmp, "Mock_Category", "B0MOCK0000", date(2023, 5, 1))
    write_json(path, {"tóm_tắt": "pin quá nóng"})
    assert read_json(path) == {"tóm_tắt": "pin quá nóng"}
    assert "pin quá nóng" in path.read_text(encoding="utf-8")  # không bị escape, đọc bằng UTF-8

    audit = AuditLog("preprocess")
    audit.record("dedup", 100, 90)
    audit.record("spam", 90, 85, flagged=5)
    assert audit.to_dataframe()["removed"].to_list() == [10, 5]
    out = audit.write(cfg_tmp.path("data_interim") / "audit")
    assert json.loads(out.read_text(encoding="utf-8").splitlines()[0])["step"] == "dedup"
