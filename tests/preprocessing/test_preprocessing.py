import json
from datetime import UTC, date, datetime, timedelta

import polars as pl

from brandsentinel.core.ids import review_id
from brandsentinel.core.logging import AuditLog
from brandsentinel.core.schemas import validate
from brandsentinel.core.types import Table
from brandsentinel.features.reliability import make_reliability_features
from brandsentinel.pipeline import run_preprocess
from brandsentinel.preprocessing.aggregate import make_daily_agg
from brandsentinel.preprocessing.dedup import deduplicate_reviews
from brandsentinel.preprocessing.loader import load_reviews
from brandsentinel.preprocessing.normalize import normalize_reviews
from brandsentinel.preprocessing.spam import flag_spam_reviews
from brandsentinel.testing.mock_data import make_daily_agg as make_mock_daily_agg
from scripts.download_amazon import _accumulate, _select_skus


def test_load_real_category_shape_and_missing_optional_columns():
    timestamp_ms = 1_704_067_200_123
    raw = pl.DataFrame(
        {
            "parent_asin": ["B0BABY001"],
            "timestamp": [timestamp_ms],
            "rating": [4.0],
            "text": ["Works very well"],
        }
    )

    clean = load_reviews(raw, "Baby_Products")

    validate(Table.CLEAN_REVIEWS, clean)
    assert clean["category"].to_list() == ["Baby_Products"]
    assert clean["review_id"].item() == review_id("B0BABY001", None, timestamp_ms)
    assert clean["ts"].dtype == pl.Datetime("us", "UTC")
    assert clean["user_id"].null_count() == 1
    assert clean["verified_purchase"].null_count() == 1
    assert clean["date"].item() == clean["ts"].dt.date().item()


def test_normalize_and_deduplicate_exact_repeats(tmp_path):
    timestamp_ms = 1_704_067_200_123
    raw = pl.DataFrame(
        {
            "parent_asin": ["B0BABY001", "B0BABY001"],
            "user_id": ["USER1", "USER1"],
            "timestamp": [timestamp_ms + 1000, timestamp_ms],
            "rating": [4, 4],
            "text": ["<b>GREATTTTT product 👍</b>", "greattttt product 👍"],
            "verified_purchase": [True, True],
        }
    )
    clean = load_reviews(raw, "Baby_Products")
    audit = AuditLog("preprocess")

    normalized = normalize_reviews(clean, audit=audit)
    deduplicated = deduplicate_reviews(normalized, audit=audit)

    assert normalized["text_norm"].to_list() == [
        "greatttt product emoji_thumbs_up",
        "greatttt product emoji_thumbs_up",
    ]
    assert deduplicated.height == 1
    assert deduplicated["ts"].item() == clean["ts"].min()
    assert deduplicated["text_raw"].item() == clean["text_raw"][1]
    assert [step["step"] for step in audit.steps] == ["normalize", "deduplicate_exact"]
    assert audit.steps[-1]["removed"] == 1
    assert len(audit.write(tmp_path).read_text(encoding="utf-8").splitlines()) == 2


def test_daily_aggregate_matches_mock_implementation(mock_tables):
    clean = mock_tables[Table.CLEAN_REVIEWS]

    actual = make_daily_agg(clean)
    expected = make_mock_daily_agg(clean)

    validate(Table.DAILY_AGG, actual)
    assert actual.equals(expected)


def test_sku_selection_requires_history_and_weekly_density():
    stats = {}
    start = date(2023, 1, 1)
    records = [
        {
            "parent_asin": "MATURE",
            "timestamp": int(
                datetime.combine(start + timedelta(days=day), datetime.min.time(), UTC).timestamp()
                * 1000
            ),
        }
        for day in range(35)
    ]
    sparse_end = datetime.combine(
        start + timedelta(days=34), datetime.min.time(), UTC
    ).timestamp()
    records.extend(
        [
            {
                "parent_asin": "SPARSE",
                "timestamp": int(
                    datetime.combine(start, datetime.min.time(), UTC).timestamp() * 1000
                ),
            },
            {
                "parent_asin": "SPARSE",
                "timestamp": int(sparse_end * 1000),
            },
        ]
    )
    for record in records:
        _accumulate(stats, record)

    selected = _select_skus(
        stats,
        min_history_days=30,
        min_total_reviews=30,
        min_median_weekly_reviews=5,
    )

    assert list(selected) == ["MATURE"]
    assert selected["MATURE"]["history_days"] == 35


def test_spam_flagging_keeps_reviews_and_records_rule_counts(cfg):
    raw = pl.DataFrame(
        {
            "parent_asin": ["SKU1"] * 3,
            "user_id": ["USER1"] * 3,
            "timestamp": [1_704_067_200_000, 1_704_067_201_000, 1_704_067_202_000],
            "rating": [5, 5, 5],
            "text": ["useful review text"] * 3,
            "verified_purchase": [True] * 3,
        }
    )
    clean = normalize_reviews(load_reviews(raw, "Baby_Products"))
    audit = AuditLog("preprocess")
    flagged = flag_spam_reviews(clean, config=cfg.default.preprocessing.spam, audit=audit)

    assert flagged.height == 3
    assert flagged["is_spam"].to_list() == [False, False, False]
    assert audit.steps[0]["n_out"] == 3


def test_reliability_features_are_daily_per_sku(mock_tables):
    tables = mock_tables
    features = make_reliability_features(
        tables[Table.DAILY_AGG],
        tables[Table.NLP_FEATURES],
        tables[Table.CLEAN_REVIEWS],
    )

    assert set(features.filter(pl.col("indicator_id") == "I10")["feature"].unique()) == {
        "mismatch_rate"
    }
    assert set(features.filter(pl.col("indicator_id") == "I11")["feature"].unique()) == {
        "verified_ratio_all"
    }
    assert features.height == 2 * tables[Table.DAILY_AGG].height
    assert features.filter(pl.col("indicator_id") == "I11")["raw_value"].drop_nulls().is_between(0, 1).all()


def test_preprocess_category_writes_filtered_tables_and_audit(tmp_path, cfg):
    start = date(2023, 1, 1)
    records = []
    for day in range(35):
        records.append(
            {
                "parent_asin": "MATURE",
                "user_id": f"USER{day}",
                "timestamp": int(
                    datetime.combine(start + timedelta(days=day), datetime.min.time(), UTC).timestamp()
                    * 1000
                ),
                "rating": 5,
                "text": "Useful product review",
                "verified_purchase": True,
            }
        )
    records.append(
        {
            "parent_asin": "SPARSE",
            "user_id": "SPARSE_USER",
            "timestamp": int(datetime.combine(start, datetime.min.time(), UTC).timestamp() * 1000),
            "rating": 5,
            "text": "Useful product review",
            "verified_purchase": True,
        }
    )
    raw_path = tmp_path / "Baby_Products.jsonl"
    pl.DataFrame(records).write_ndjson(raw_path)
    selection = cfg.default.data.sku_selection.model_copy(
        update={
            "min_history_days": 30,
            "min_total_reviews": 30,
            "min_median_weekly_reviews": 5,
        }
    )
    data = cfg.default.data.model_copy(update={"sku_selection": selection})
    default = cfg.default.model_copy(update={"data": data})
    test_cfg = cfg.model_copy(update={"root": tmp_path, "default": default})

    clean, daily, audit_path = run_preprocess(
        "Baby_Products", source=raw_path, config=test_cfg
    )

    assert clean["sku"].unique().to_list() == ["MATURE"]
    assert daily["sku"].unique().to_list() == ["MATURE"]
    assert audit_path is not None and audit_path.exists()
    audit_steps = [
        json.loads(line)["step"] for line in audit_path.read_text(encoding="utf-8").splitlines()
    ]
    assert audit_steps == ["normalize", "deduplicate_exact", "flag_spam", "select_eligible_skus"]