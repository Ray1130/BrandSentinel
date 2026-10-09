"""Integration tests for full NLP pipeline orchestration (Phase 3A)."""

import polars as pl
import pytest

from brandsentinel.core.schemas import validate
from brandsentinel.core.types import Table
from brandsentinel.features.nlp.pipeline import compute_nlp_features
from brandsentinel.testing.mock_data import make_clean_reviews


@pytest.fixture(scope="module")
def mock_clean(cfg):
    """Lấy lát cắt mẫu 30 dòng từ mock clean_reviews để test integration nhanh chóng."""
    return make_clean_reviews(seed=42).head(30)


def test_full_pipeline_on_mock(mock_clean, cfg):
    """1. Test that compute_nlp_features runs successfully on mock clean_reviews."""
    res = compute_nlp_features(mock_clean, cfg=cfg)
    assert isinstance(res, pl.DataFrame)
    assert res.height == mock_clean.height


def test_row_count_and_review_id_preservation(mock_clean, cfg):
    """2 & 3. Test input row count == output row count, review_id unique and preserved."""
    res = compute_nlp_features(mock_clean, cfg=cfg)

    assert res.height == mock_clean.height
    assert res["review_id"].to_list() == mock_clean["review_id"].to_list()
    assert res["review_id"].n_unique() == mock_clean.height
    assert res["category"].to_list() == mock_clean["category"].to_list()
    assert res["date"].to_list() == mock_clean["date"].to_list()


def test_sentiment_and_mismatch_outputs(mock_clean, cfg):
    """4 & 5. Test sentiment_score and mismatch columns exist and are well-formed."""
    res = compute_nlp_features(mock_clean, cfg=cfg)

    assert "sentiment_score" in res.columns
    assert "mismatch" in res.columns
    assert res["sentiment_score"].dtype == pl.Float32
    assert res["mismatch"].dtype == pl.Boolean

    # sentiment_score must be in [-1.0, 1.0]
    scores = res["sentiment_score"].drop_nulls().to_list()
    assert all(-1.0 <= s <= 1.0 for s in scores)


def test_risk_outputs_layer1_and_layer2(mock_clean, cfg):
    """6 & 7. Test risk_hit, risk_terms, and risk_sim columns exist and follow contract."""
    res = compute_nlp_features(mock_clean, cfg=cfg)

    assert "risk_hit" in res.columns
    assert "risk_terms" in res.columns
    assert "risk_sim" in res.columns

    assert res["risk_hit"].dtype == pl.Boolean
    assert res["risk_terms"].dtype == pl.List(pl.String)
    assert res["risk_sim"].dtype == pl.Float32

    # Semantic rule: risk_hit is False -> risk_terms must be null
    false_hits = res.filter(~pl.col("risk_hit"))
    assert false_hits["risk_terms"].null_count() == false_hits.height

    # risk_sim must be in [-1.0, 1.0] when not null
    sims = res["risk_sim"].drop_nulls().to_list()
    assert all(-1.0 <= s <= 1.0 for s in sims)


def test_aspect_outputs_and_semantic_rule(mock_clean, cfg):
    """8 & 9. Test aspect, aspect_sentiment, and aspect null -> aspect_sentiment null."""
    res = compute_nlp_features(mock_clean, cfg=cfg)

    assert "aspect" in res.columns
    assert "aspect_sentiment" in res.columns

    assert res["aspect"].dtype == pl.String
    assert res["aspect_sentiment"].dtype == pl.Float32

    # Semantic rule: aspect is null -> aspect_sentiment must be null
    null_aspects = res.filter(pl.col("aspect").is_null())
    assert null_aspects["aspect_sentiment"].null_count() == null_aspects.height


def test_schema_validation(mock_clean, cfg):
    """10. Test that output passes Table.NLP_FEATURES validate() strictly."""
    res = compute_nlp_features(mock_clean, cfg=cfg, validate_output=True)
    validated = validate(Table.NLP_FEATURES, res)
    assert validated.height == mock_clean.height


def test_deterministic_output(mock_clean, cfg):
    """11. Test that pipeline execution is deterministic."""
    slice_clean = mock_clean.head(10)
    res1 = compute_nlp_features(slice_clean, cfg=cfg)
    res2 = compute_nlp_features(slice_clean, cfg=cfg)

    assert res1["sentiment_score"].to_list() == res2["sentiment_score"].to_list()
    assert res1["risk_hit"].to_list() == res2["risk_hit"].to_list()
    assert res1["risk_sim"].to_list() == res2["risk_sim"].to_list()
    assert res1["aspect"].to_list() == res2["aspect"].to_list()


def test_empty_dataframe(cfg):
    """12. Test pipeline behavior on empty clean_reviews."""
    empty_clean = pl.DataFrame(
        schema={
            "review_id": pl.String,
            "category": pl.String,
            "date": pl.Date,
            "text_norm": pl.String,
            "rating": pl.Int8,
        }
    )
    res = compute_nlp_features(empty_clean, cfg=cfg, validate_output=True)
    assert res.is_empty()
    assert list(res.columns) == [
        "review_id",
        "category",
        "date",
        "sentiment_score",
        "risk_hit",
        "risk_terms",
        "risk_sim",
        "aspect",
        "aspect_sentiment",
        "mismatch",
    ]


def test_duplicate_review_id_rejected(cfg):
    """13. Test that duplicate review_id raises ValueError."""
    dup_df = pl.DataFrame(
        {
            "review_id": ["DUP_01", "DUP_01"],
            "category": ["Baby_Products", "Baby_Products"],
            "date": [pl.date(2023, 1, 1), pl.date(2023, 1, 2)],
            "text_norm": ["Great product", "Awful product"],
            "rating": [5, 1],
        }
    )
    with pytest.raises(ValueError, match="trùng lặp"):
        compute_nlp_features(dup_df, cfg=cfg)
