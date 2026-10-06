from datetime import UTC, date, datetime, timedelta

import polars as pl

from brandsentinel.core.ids import review_id
from brandsentinel.core.logging import AuditLog
from brandsentinel.core.schemas import validate
from brandsentinel.core.types import Table
from brandsentinel.preprocessing.aggregate import make_daily_agg
from brandsentinel.preprocessing.dedup import deduplicate_reviews
from brandsentinel.preprocessing.loader import load_reviews
from brandsentinel.preprocessing.normalize import normalize_reviews
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