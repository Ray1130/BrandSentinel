"""Unit tests for aspect.py (I9)."""

from pathlib import Path

import polars as pl
import pytest

from brandsentinel.core.schemas import validate
from brandsentinel.core.types import Table
from brandsentinel.features.nlp.aspect import AspectExtractor, compute_aspect
from brandsentinel.testing.mock_data import make_clean_reviews


@pytest.fixture(scope="module")
def extractor(cfg):
    return AspectExtractor(cfg=cfg)


def test_quality_positive_and_negative(extractor):
    df_pos = pl.DataFrame(
        {
            "review_id": ["Q_POS"],
            "text_norm": ["The build is extremely durable and the materials feel high quality."],
        }
    )
    res_pos = extractor.extract_aspects(df_pos)
    assert res_pos["aspect"][0] == "quality"
    assert res_pos["aspect_sentiment"][0] > 0.4

    df_neg = pl.DataFrame(
        {
            "review_id": ["Q_NEG"],
            "text_norm": ["The plastic zipper snapped on day one, cheap and defective build."],
        }
    )
    res_neg = extractor.extract_aspects(df_neg)
    assert res_neg["aspect"][0] == "quality"
    assert res_neg["aspect_sentiment"][0] < -0.4


def test_delivery_positive_and_negative(extractor):
    df_pos = pl.DataFrame(
        {
            "review_id": ["D_POS"],
            "text_norm": ["Fast shipping and the package arrived early with great packaging."],
        }
    )
    res_pos = extractor.extract_aspects(df_pos)
    assert res_pos["aspect"][0] == "delivery"
    assert res_pos["aspect_sentiment"][0] > 0.4

    df_neg = pl.DataFrame(
        {
            "review_id": ["D_NEG"],
            "text_norm": ["Delivery was late and the box arrived smashed in transit."],
        }
    )
    res_neg = extractor.extract_aspects(df_neg)
    assert res_neg["aspect"][0] == "delivery"
    assert res_neg["aspect_sentiment"][0] < -0.4


def test_safety_positive_and_negative(extractor):
    df_pos = pl.DataFrame(
        {
            "review_id": ["S_POS"],
            "text_norm": ["Heater has auto-shutoff and stays cool, very safe for toddlers."],
        }
    )
    res_pos = extractor.extract_aspects(df_pos)
    assert res_pos["aspect"][0] == "safety"
    assert res_pos["aspect_sentiment"][0] > 0.4

    df_neg = pl.DataFrame(
        {
            "review_id": ["S_NEG"],
            "text_norm": ["The battery started to overheat and caught fire, severe hazard!"],
        }
    )
    res_neg = extractor.extract_aspects(df_neg)
    assert res_neg["aspect"][0] == "safety"
    assert res_neg["aspect_sentiment"][0] < -0.5


def test_refund_positive_and_negative(extractor):
    df_pos = pl.DataFrame(
        {
            "review_id": ["R_POS"],
            "text_norm": ["Seller issued a prompt refund, great customer service and support."],
        }
    )
    res_pos = extractor.extract_aspects(df_pos)
    assert res_pos["aspect"][0] == "refund"
    assert res_pos["aspect_sentiment"][0] > 0.4

    df_neg = pl.DataFrame(
        {
            "review_id": ["R_NEG"],
            "text_norm": ["Customer support refused my return and denied my refund request."],
        }
    )
    res_neg = extractor.extract_aspects(df_neg)
    assert res_neg["aspect"][0] == "refund"
    assert res_neg["aspect_sentiment"][0] < -0.4


def test_no_aspect_returns_null(extractor):
    df = pl.DataFrame(
        {
            "review_id": ["N1", "N2", "N3", "N4"],
            "text_norm": [None, "", "   ", "ok, nothing special"],
        }
    )
    res = extractor.extract_aspects(df)
    assert res.height == 4
    assert res["aspect"].to_list() == [None, None, None, None]
    assert res["aspect_sentiment"].to_list() == [None, None, None, None]


def test_multi_aspect_dominant_ranking(extractor):
    # Case 1: 2 câu về delivery, 1 câu về quality -> dominant = delivery
    text_delivery_wins = "The shipping was delayed for weeks. The package arrived torn and damaged. But the screen is nice."
    asp, _ = extractor.find_dominant_aspect(text_delivery_wins)
    assert asp == "delivery"

    # Case 2: Cùng 1 câu, nhưng có cả 'safety' (overheat) và 'quality' (battery)
    # Theo tie-break an toàn: safety > quality -> dominant = safety
    text_tie_break = "The battery started to overheat rapidly."
    asp, _ = extractor.find_dominant_aspect(text_tie_break)
    assert asp == "safety"


def test_aspect_sentiment_isolates_aspect_sentences(extractor):
    # Review có 2 vế:
    # Câu 1: Quality xuất sắc ("The fabric is super soft, gorgeous material.")
    # Câu 2: Delivery tệ ("Delivery was horribly late and courier was rude.")
    # Ở đây cả 2 aspect đều có 1 câu, nhưng delivery (priority 2) > quality (priority 1)
    text = "The fabric is super soft, gorgeous material. However, delivery was horribly late and courier was rude."
    df = pl.DataFrame({"review_id": ["REV_MIXED"], "text_norm": [text]})
    res = extractor.extract_aspects(df)

    # Dominant aspect phải là delivery (do tie-break ưu tiên delivery hơn quality)
    assert res["aspect"][0] == "delivery"
    # aspect_sentiment phải được tính trên câu delivery (tiêu cực), không bị câu khen quality kéo lên
    assert res["aspect_sentiment"][0] < 0.0


def test_batch_processing_and_preservation(extractor):
    ids = [f"REV_{i:03d}" for i in range(50)]
    texts = [
        "fast delivery, arrived on time" if i % 2 == 0 else "cheap broke zipper"
        for i in range(50)
    ]
    df = pl.DataFrame({"review_id": ids, "text_norm": texts})
    res = extractor.extract_aspects(df)

    assert res.height == 50
    assert res["review_id"].to_list() == ids
    assert (res["aspect_sentiment"].is_not_null()).all()


def test_schema_integration(cfg, extractor):
    clean = make_clean_reviews(n_skus=2, days=5, seed=42)
    aspect_df = compute_aspect(clean, cfg=cfg, extractor=extractor)

    assert aspect_df.height == clean.height
    assert (aspect_df["review_id"] == clean["review_id"]).all()

    # Ghép thử vào schema đầy đủ của Table.NLP_FEATURES để validate
    joined = clean.select("review_id", "category", "date").join(
        aspect_df, on="review_id", how="left"
    )
    nlp_full = joined.with_columns(
        pl.lit(0.0, dtype=pl.Float32).alias("sentiment_score"),
        pl.lit(False).alias("risk_hit"),
        pl.lit(None, dtype=pl.List(pl.String)).alias("risk_terms"),
        pl.lit(None, dtype=pl.Float32).alias("risk_sim"),
        pl.lit(False).alias("mismatch"),
    )

    # Validate chính thức với Table.NLP_FEATURES
    validate(Table.NLP_FEATURES, nlp_full)
