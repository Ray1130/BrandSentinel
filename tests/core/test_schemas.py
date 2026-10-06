import polars as pl
import pytest

from brandsentinel.core.ids import review_id
from brandsentinel.core.schemas import (
    POLARS_SCHEMAS,
    DataContractError,
    conform,
    empty_table,
    validate,
)
from brandsentinel.core.types import Level, Table


@pytest.mark.parametrize("table", list(Table))
def test_mock_tables_valid(mock_tables, table):
    validate(table, mock_tables[table])


@pytest.mark.parametrize("table", list(Table))
def test_empty_table_valid(table):
    df = empty_table(table)
    assert df.schema == pl.Schema(POLARS_SCHEMAS[table])
    validate(table, df)


def test_mock_scenarios(mock_tables):
    alerts = mock_tables[Table.ALERTS]
    last = alerts["window_end"].max()
    tail = alerts.filter(pl.col("window_end") == last)
    level = dict(zip(tail["sku"], tail["level"], strict=True))
    assert level["B0MOCK0000"] == Level.HIGH  # SKU khủng hoảng
    assert tail.filter(pl.col("sku") == "B0MOCK0001")["verify_flag"].item()  # review bombing
    normal = alerts.filter(~pl.col("sku").is_in(["B0MOCK0000", "B0MOCK0001"]))
    assert (normal["level"] == Level.HIGH).sum() == 0  # SKU bình thường không lên HIGH


def test_review_id_stable_and_unique(mock_tables):
    assert review_id("B1", "U1", 1) == review_id("B1", "U1", 1)
    assert review_id("B1", "U1", 1) != review_id("B1", "U1", 2)
    assert review_id("B1", None, 1) == review_id("B1", "", 1)
    assert len(review_id("B1", "U1", 1)) == 16
    ids = mock_tables[Table.CLEAN_REVIEWS]["review_id"]
    assert ids.n_unique() == len(ids)


def test_rating_out_of_range(mock_tables):
    df = mock_tables[Table.CLEAN_REVIEWS].with_columns(pl.lit(6).cast(pl.Int8).alias("rating"))
    with pytest.raises(DataContractError):
        validate(Table.CLEAN_REVIEWS, df)


def test_duplicate_key_rejected(mock_tables):
    df = mock_tables[Table.CLEAN_REVIEWS]
    with pytest.raises(DataContractError):
        validate(Table.CLEAN_REVIEWS, pl.concat([df, df.head(1)]))


def test_extra_column_rejected(mock_tables):
    df = mock_tables[Table.DAILY_AGG].with_columns(pl.lit(1).alias("oops"))
    with pytest.raises(DataContractError):
        validate(Table.DAILY_AGG, df)


def test_naive_timestamp_rejected(mock_tables):
    df = mock_tables[Table.CLEAN_REVIEWS].with_columns(pl.col("ts").dt.replace_time_zone(None))
    with pytest.raises(DataContractError):
        validate(Table.CLEAN_REVIEWS, df)


def test_date_must_match_ts(mock_tables):
    df = mock_tables[Table.CLEAN_REVIEWS].with_columns(pl.col("date") + pl.duration(days=1))
    with pytest.raises(DataContractError, match="date"):
        validate(Table.CLEAN_REVIEWS, df)


def test_daily_agg_histogram_consistency(mock_tables):
    df = mock_tables[Table.DAILY_AGG].with_columns((pl.col("n") + 1).cast(pl.Int32).alias("n"))
    with pytest.raises(DataContractError, match="n1"):
        validate(Table.DAILY_AGG, df)


def test_unknown_feature_rejected(mock_tables):
    df = mock_tables[Table.FEATURE_SERIES].with_columns(pl.lit("made_up").alias("feature"))
    with pytest.raises(DataContractError):
        validate(Table.FEATURE_SERIES, df)


def test_nan_rejected_use_null(mock_tables):
    df = mock_tables[Table.FEATURE_SERIES].with_columns(
        pl.when(pl.int_range(pl.len()) == 0)
        .then(float("nan"))
        .otherwise(pl.col("raw_value"))
        .alias("raw_value")
    )
    with pytest.raises(DataContractError, match="NaN"):
        validate(Table.FEATURE_SERIES, df)


def test_alert_group_outside_contract(mock_tables):
    df = mock_tables[Table.ALERTS]
    bad = df.with_columns(pl.Series("groups_active", [["flag"]] * df.height))
    with pytest.raises(DataContractError, match="groups_active"):
        validate(Table.ALERTS, bad)


def test_conform_casts_and_rejects_extra(mock_tables):
    df = mock_tables[Table.DAILY_AGG]
    loose = df.with_columns(pl.col("n").cast(pl.Int64)).select(reversed(df.columns))
    assert conform(Table.DAILY_AGG, loose).schema == pl.Schema(POLARS_SCHEMAS[Table.DAILY_AGG])
    with pytest.raises(DataContractError):
        conform(Table.DAILY_AGG, df.with_columns(pl.lit(1).alias("extra")))
