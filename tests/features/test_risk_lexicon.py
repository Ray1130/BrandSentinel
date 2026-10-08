"""Unit tests for risk_lexicon.py (I8 - Stage 1)."""

from pathlib import Path

import polars as pl
import pytest

from brandsentinel.core.schemas import validate
from brandsentinel.core.types import Table
from brandsentinel.features.nlp.risk_lexicon import RiskLexiconDetector, compute_risk_lexicon
from brandsentinel.testing.mock_data import make_clean_reviews


@pytest.fixture(scope="module")
def detector(cfg):
    return RiskLexiconDetector(cfg=cfg)


@pytest.mark.parametrize(
    ("text", "expected_hit", "expected_terms"),
    [
        # 1. Product name false positive
        ("Love my Fire TV Stick", False, None),
        ("Bought an Amazon Fire HD tablet for kids", False, None),
        ("Got this during the fire sale weekend", False, None),
        ("Fire department safety demonstration", False, None),
        # 2. True fire / heat
        ("It caught fire", True, ["caught fire"]),
        ("The battery pack caught fire overnight", True, ["caught fire"]),
        ("Charger began to overheat rapidly", True, ["overheat"]),
        ("Noticeable burning smell from the motor", True, ["burning smell"]),
        # 3. Negation cases
        ("No fire hazard at all", False, None),
        ("Never overheats even during intense gaming", False, None),
        ("Not a fake, it's genuine", False, None),
        ("Zero burning smell and no smoke", False, None),
        ("Hypoallergenic cream, caused no rash", False, None),
        ("Verified not toxic and safe", False, None),
        # 4. Counterfeit / Fake
        ("This is a fake", True, ["fake"]),
        ("Received a dangerous counterfeit adapter", True, ["fake"]),
        # 5. Health / Allergy / Injury
        ("Baby got a severe allergic rash", True, ["allergic", "rash"]),
        ("Sharp edge caused deep finger injury", True, ["injury"]),
        ("Fumes were toxic and caused choking", True, ["toxic"]),
        # 6. Legal / Recall
        ("Product was under safety recall notice", True, ["recall"]),
        ("Customers are filing a class action lawsuit", True, ["lawsuit"]),
        # 7. Case sensitivity & Word boundary
        ("THE MOTOR OVERHEATED AND CAUGHT FIRE", True, ["overheat", "caught fire"]),
        ("The car did not crash today", False, None),  # 'crash' contains 'rash', should not match
    ],
)
def test_risk_detection_cases(detector, text, expected_hit, expected_terms):
    hit, terms = detector.detect_text(text)
    assert hit is expected_hit, f"Failed hit for: {text!r}"
    if expected_terms is None:
        assert terms is None or len(terms) == 0, f"Expected no terms, got {terms} for: {text!r}"
    else:
        assert terms is not None, f"Expected {expected_terms}, got None for: {text!r}"
        for t in expected_terms:
            assert t in terms, f"Expected term {t} not found in {terms} for: {text!r}"


def test_clause_breaker_prevents_negation_leak(detector):
    # 'not' nằm ở câu trước dấu chấm, không được phủ định 'caught fire' ở câu sau
    text = "It was not very pretty. Then it caught fire."
    hit, terms = detector.detect_text(text)
    assert hit is True
    assert "caught fire" in terms


def test_empty_and_null_text(detector):
    assert detector.detect_text(None) == (False, None)
    assert detector.detect_text("") == (False, None)
    assert detector.detect_text("    ") == (False, None)


def test_compute_risk_dataframe(detector):
    df = pl.DataFrame(
        {
            "review_id": ["R1", "R2", "R3"],
            "text_norm": ["it caught fire", "never overheats", None],
        }
    )
    res = detector.compute_risk(df)
    assert res.height == 3
    assert res["risk_hit"].to_list() == [True, False, False]
    assert res["risk_terms"][0].to_list() == ["caught fire"]
    assert res["risk_terms"][1] is None
    assert res["risk_terms"][2] is None


def test_schema_integration(cfg, detector):
    clean = make_clean_reviews(n_skus=2, days=5, seed=42)
    risk_df = compute_risk_lexicon(clean, cfg=cfg, detector=detector)

    assert risk_df.height == clean.height
    assert (risk_df["review_id"] == clean["review_id"]).all()

    # Ghép thử vào schema đầy đủ của Table.NLP_FEATURES để validate
    joined = clean.select("review_id", "category", "date").join(
        risk_df, on="review_id", how="left"
    )
    nlp_full = joined.with_columns(
        pl.lit(0.0, dtype=pl.Float32).alias("sentiment_score"),
        pl.lit(None, dtype=pl.Float32).alias("risk_sim"),
        pl.lit(None, dtype=pl.String).alias("aspect"),
        pl.lit(None, dtype=pl.Float32).alias("aspect_sentiment"),
        pl.lit(False).alias("mismatch"),
    )

    # Validate chính thức
    validate(Table.NLP_FEATURES, nlp_full)
