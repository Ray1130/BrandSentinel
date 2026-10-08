"""Unit tests and integration validation for sentiment.py."""

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from brandsentinel.core.schemas import validate
from brandsentinel.core.types import Table
from brandsentinel.features.nlp.sentiment import SentimentAnalyzer, compute_sentiment
from brandsentinel.testing.mock_data import make_clean_reviews

FIXTURE_PATH = Path("tests/fixtures/sample_reviews.csv")


@pytest.fixture(scope="module")
def analyzer(cfg):
    return SentimentAnalyzer(cfg=cfg)


def test_clear_positive(analyzer):
    df = pl.DataFrame(
        {
            "review_id": ["REV_POS_1"],
            "rating": [5],
            "text_norm": ["absolutely fantastic product, works perfectly and high quality"],
        },
        schema={"review_id": pl.String, "rating": pl.Int8, "text_norm": pl.String},
    )
    res = analyzer.compute_sentiment(df)
    assert res.height == 1
    assert res["sentiment_score"][0] > 0.5
    assert res["mismatch"][0] is False


def test_clear_negative(analyzer):
    df = pl.DataFrame(
        {
            "review_id": ["REV_NEG_1"],
            "rating": [1],
            "text_norm": ["terrible item, broke on the first day, complete waste of money"],
        },
        schema={"review_id": pl.String, "rating": pl.Int8, "text_norm": pl.String},
    )
    res = analyzer.compute_sentiment(df)
    assert res.height == 1
    assert res["sentiment_score"][0] < -0.5
    assert res["mismatch"][0] is False


def test_clear_neutral(analyzer):
    df = pl.DataFrame(
        {
            "review_id": ["REV_NEU_1"],
            "rating": [3],
            "text_norm": ["standard product, does the job, nothing special"],
        },
        schema={"review_id": pl.String, "rating": pl.Int8, "text_norm": pl.String},
    )
    res = analyzer.compute_sentiment(df)
    assert res.height == 1
    # Score should be close to 0
    assert abs(res["sentiment_score"][0]) < 0.5
    assert res["mismatch"][0] is False


def test_high_rating_negative_text_mismatch(analyzer):
    # Rating 5 but very negative text
    df = pl.DataFrame(
        {
            "review_id": ["REV_MIS_1"],
            "rating": [5],
            "text_norm": ["horrible quality, arrived smashed and completely unusable"],
        },
        schema={"review_id": pl.String, "rating": pl.Int8, "text_norm": pl.String},
    )
    res = analyzer.compute_sentiment(df)
    assert res["sentiment_score"][0] < -0.5
    assert res["mismatch"][0] is True


def test_low_rating_positive_text_mismatch(analyzer):
    # Rating 1 but very positive text
    df = pl.DataFrame(
        {
            "review_id": ["REV_MIS_2"],
            "rating": [1],
            "text_norm": ["best purchase ever, wonderful craftsmanship, highly recommend"],
        },
        schema={"review_id": pl.String, "rating": pl.Int8, "text_norm": pl.String},
    )
    res = analyzer.compute_sentiment(df)
    assert res["sentiment_score"][0] > 0.5
    assert res["mismatch"][0] is True


def test_rating_3_never_mismatch(analyzer):
    # Even with strong text, rating 3 should not be flagged as mismatch
    df = pl.DataFrame(
        {
            "review_id": ["REV_R3_POS", "REV_R3_NEG"],
            "rating": [3, 3],
            "text_norm": [
                "absolutely wonderful best product in the world",
                "horrible worst trash completely broken garbage",
            ],
        },
        schema={"review_id": pl.String, "rating": pl.Int8, "text_norm": pl.String},
    )
    res = analyzer.compute_sentiment(df)
    assert res["mismatch"][0] is False
    assert res["mismatch"][1] is False


def test_empty_and_null_text(analyzer):
    df = pl.DataFrame(
        {
            "review_id": ["REV_EMPTY", "REV_NULL", "REV_SPACES"],
            "rating": [5, 1, 3],
            "text_norm": ["", None, "   "],
        },
        schema={"review_id": pl.String, "rating": pl.Int8, "text_norm": pl.String},
    )
    res = analyzer.compute_sentiment(df)
    assert res.height == 3
    assert res["sentiment_score"].to_list() == [None, None, None]
    assert res["mismatch"].to_list() == [None, None, None]


def test_long_text_truncation(analyzer):
    long_text = "great excellent wonderful " * 100
    df = pl.DataFrame(
        {
            "review_id": ["REV_LONG"],
            "rating": [5],
            "text_norm": [long_text],
        },
        schema={"review_id": pl.String, "rating": pl.Int8, "text_norm": pl.String},
    )
    res = analyzer.compute_sentiment(df)
    assert res.height == 1
    assert res["sentiment_score"][0] > 0.5


def test_batch_processing_and_preservation(analyzer):
    ids = [f"REV_{i:03d}" for i in range(75)]
    texts = ["good item" if i % 2 == 0 else "bad product" for i in range(75)]
    ratings = [5 if i % 2 == 0 else 1 for i in range(75)]

    df = pl.DataFrame(
        {"review_id": ids, "rating": ratings, "text_norm": texts},
        schema={"review_id": pl.String, "rating": pl.Int8, "text_norm": pl.String},
    )
    res = analyzer.compute_sentiment(df)

    assert res.height == 75
    assert res["review_id"].to_list() == ids
    assert (res["sentiment_score"] >= -1.0).all() and (res["sentiment_score"] <= 1.0).all()


def test_deterministic_output(analyzer):
    df = pl.DataFrame(
        {
            "review_id": ["REV_DET_1", "REV_DET_2"],
            "rating": [5, 1],
            "text_norm": ["love this very much", "hate it so bad"],
        },
        schema={"review_id": pl.String, "rating": pl.Int8, "text_norm": pl.String},
    )
    res1 = analyzer.compute_sentiment(df)
    res2 = analyzer.compute_sentiment(df)

    assert res1["sentiment_score"].to_list() == res2["sentiment_score"].to_list()
    assert res1["mismatch"].to_list() == res2["mismatch"].to_list()


def test_schema_integration_with_clean_reviews(cfg, analyzer):
    clean = make_clean_reviews(n_skus=2, days=5, seed=42)
    nlp_sentiment = compute_sentiment(clean, cfg=cfg, analyzer=analyzer)

    assert nlp_sentiment.height == clean.height
    assert (nlp_sentiment["review_id"] == clean["review_id"]).all()

    # Ghép thử vào schema đầy đủ của NLP_FEATURES để validate
    joined = clean.select("review_id", "category", "date").join(
        nlp_sentiment, on="review_id", how="left"
    )
    nlp_full = joined.with_columns(
        pl.lit(False).alias("risk_hit"),
        pl.lit(None, dtype=pl.List(pl.String)).alias("risk_terms"),
        pl.lit(None, dtype=pl.Float32).alias("risk_sim"),
        pl.lit(None, dtype=pl.String).alias("aspect"),
        pl.lit(None, dtype=pl.Float32).alias("aspect_sentiment"),
    )

    # Validate bằng schema chính thức của repo
    validate(Table.NLP_FEATURES, nlp_full)
