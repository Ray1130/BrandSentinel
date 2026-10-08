"""Unit tests for embedder.py (I8 Layer 2 - Semantic Embedding)."""

import numpy as np
import polars as pl
import pytest

from brandsentinel.core.schemas import validate
from brandsentinel.core.types import Table
from brandsentinel.features.nlp.embedder import (
    DEFAULT_RISK_PROTOTYPES,
    RiskEmbedder,
    compute_embedding_similarity,
    compute_i8_combined,
    get_risk_embedder,
)
from brandsentinel.testing.mock_data import make_clean_reviews


@pytest.fixture(scope="module")
def embedder(cfg):
    return get_risk_embedder(cfg=cfg)


def test_embedder_initialization(embedder):
    """1. Test model initialization, dimension, and prototypes."""
    assert embedder is not None
    assert embedder.dimension == 384
    assert set(embedder.proto_categories) == {"safety", "health", "counterfeit", "legal"}
    assert embedder.proto_embeddings.shape == (4, 384)
    # Prototype embeddings must be L2 normalized
    norms = np.linalg.norm(embedder.proto_embeddings, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_single_text_embedding(embedder):
    """2. Test single text embedding shape and normalization."""
    text = "The battery overheated and started smoking"
    emb = embedder.encode([text])
    assert emb.shape == (1, 384)
    assert emb.dtype == np.float32
    norm = np.linalg.norm(emb[0])
    assert pytest.approx(norm, abs=1e-5) == 1.0


def test_batch_embedding(embedder):
    """3. Test batch text embedding."""
    texts = [
        "First review text about quality",
        "Second review text about delivery",
        "Third review text about battery explosion",
    ]
    emb = embedder.encode(texts, batch_size=2)
    assert emb.shape == (3, 384)
    norms = np.linalg.norm(emb, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_deterministic_output(embedder):
    """4. Test that multiple runs produce identical embeddings."""
    text = "Consistent review text for deterministic check"
    emb1 = embedder.encode([text])
    emb2 = embedder.encode([text])
    np.testing.assert_array_equal(emb1, emb2)


def test_null_and_empty_text(embedder):
    """5. Test safe handling of null, empty, and whitespace text."""
    texts = [None, "", "   ", "Valid review text", None]
    emb = embedder.encode(texts)
    assert emb.shape == (5, 384)
    # Invalid rows must be zero vectors
    assert np.all(emb[0] == 0.0)
    assert np.all(emb[1] == 0.0)
    assert np.all(emb[2] == 0.0)
    assert pytest.approx(np.linalg.norm(emb[3]), abs=1e-5) == 1.0
    assert np.all(emb[4] == 0.0)

    # Similarities for invalid rows must be None
    sims, cat_sims = embedder.similarity_to_prototypes(texts)
    assert sims[0] is None
    assert sims[1] is None
    assert sims[2] is None
    assert isinstance(sims[3], float)
    assert sims[4] is None
    for cat in embedder.proto_categories:
        assert cat_sims[cat][0] is None
        assert isinstance(cat_sims[cat][3], float)


def test_risk_prototype_similarity(embedder):
    """6. Test similarity scores to individual category prototypes."""
    text = "The medication caused a painful skin rash and allergy"
    sims, cat_sims = embedder.similarity_to_prototypes([text])
    assert sims[0] is not None
    # Health similarity should be higher than delivery or legal
    assert cat_sims["health"][0] > cat_sims["legal"][0]


def test_high_risk_semantic_example(embedder):
    """7. Test semantic similarity is elevated on dangerous phrasing."""
    high_risk_texts = [
        "The charger erupted in flames and burned the carpet",
        "Toxic chemical fumes caused severe respiratory distress",
        "Total fake counterfeit knockoff that broke instantly",
        "Manufacturer issued a nationwide safety recall",
    ]
    sims, _ = embedder.similarity_to_prototypes(high_risk_texts)
    for text, sim in zip(high_risk_texts, sims, strict=True):
        assert sim is not None
        assert sim > 0.40, f"Expected elevated similarity for '{text}', got {sim}"


def test_non_risk_semantic_example(embedder):
    """8. Test benign text has low similarity to risk prototypes."""
    benign_texts = [
        "The baby blanket is delightfully soft and warm",
        "Fast delivery, arrived two days earlier than expected",
        "Good coffee maker, brews smooth coffee every morning",
    ]
    sims, _ = embedder.similarity_to_prototypes(benign_texts)
    for text, sim in zip(benign_texts, sims, strict=True):
        assert sim is not None
        assert sim < 0.35, f"Expected low similarity for '{text}', got {sim}"


def test_risk_sim_range(embedder):
    """10. Test that risk_sim is strictly within [-1.0, 1.0]."""
    texts = [
        "Horrible explosion and fire hazard",
        "Completely safe and nice item",
        "Ordinary item",
    ]
    sims, _ = embedder.similarity_to_prototypes(texts)
    for sim in sims:
        assert sim is not None
        assert -1.0 <= sim <= 1.0
        assert not np.isnan(sim)
        assert not np.isinf(sim)


def test_layer1_and_layer2_integration(cfg):
    """9. Test compute_i8_combined in layer1, layer2, and combined modes."""
    reviews = pl.DataFrame(
        {
            "review_id": ["REV_01", "REV_02", "REV_03"],
            "text_norm": [
                "It caught fire and burned the desk",  # L1 hit + L2 hit
                "Delightful soft cotton fabric",       # Neither L1 nor L2
                "The unit melted and sparked violently", # Semantic hazard
            ],
        }
    )

    # Mode Layer 1
    res_l1 = compute_i8_combined(reviews, cfg=cfg, mode="layer1")
    assert res_l1["risk_hit"].to_list()[0] is True
    assert res_l1["risk_hit"].to_list()[1] is False
    assert "risk_sim" in res_l1.columns

    # Mode Layer 2 (th=0.40)
    res_l2 = compute_i8_combined(reviews, cfg=cfg, threshold=0.40, mode="layer2")
    assert res_l2["risk_hit"].to_list()[0] is True
    assert res_l2["risk_hit"].to_list()[1] is False
    assert res_l2["risk_terms"].null_count() == 3

    # Mode Combined
    res_comb = compute_i8_combined(reviews, cfg=cfg, threshold=0.40, mode="combined")
    assert res_comb["risk_hit"].to_list()[0] is True
    assert res_comb["risk_hit"].to_list()[1] is False
    assert res_comb["risk_terms"].to_list()[0] == ["caught fire"]
    assert res_comb["risk_terms"].to_list()[1] is None


def test_schema_compatible_output(cfg):
    """11. Test that integrated output strictly conforms to Table.NLP_FEATURES."""
    clean = make_clean_reviews(seed=42).head(15)
    i8_df = compute_i8_combined(clean, cfg=cfg, threshold=0.40)

    # Assemble NLP_FEATURES mock table
    nlp_df = clean.select(["review_id", "category", "date"]).with_columns(
        pl.lit(0.5, dtype=pl.Float32).alias("sentiment_score"),
        pl.lit(None, dtype=pl.Boolean).alias("mismatch"),
        pl.lit("quality", dtype=pl.String).alias("aspect"),
        pl.lit(0.5, dtype=pl.Float32).alias("aspect_sentiment"),
    ).join(
        i8_df.select(["review_id", "risk_hit", "risk_terms", "risk_sim"]),
        on="review_id",
        how="left",
    )

    validated = validate(Table.NLP_FEATURES, nlp_df)
    assert validated.height == 15
    assert "risk_sim" in validated.columns
    assert validated["risk_sim"].dtype == pl.Float32
    assert validated["risk_hit"].dtype == pl.Boolean
