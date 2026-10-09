"""Unit tests for topic burst detection (Indicator I7) - Stage 1."""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from brandsentinel.features.nlp.topic import (
    compute_term_burst,
    compute_topic_window_features,
    extract_ngrams,
)


def test_extract_ngrams_basic():
    text = "The stroller has a broken wheel and poor build quality"
    ngrams = extract_ngrams(text, ngram_range=(1, 2))
    assert "stroller" in ngrams
    assert "broken" in ngrams
    assert "broken wheel" in ngrams
    assert "the" not in ngrams
    assert "has" not in ngrams


def test_extract_ngrams_empty_and_none():
    assert extract_ngrams("") == []
    assert extract_ngrams(None) == []
    assert extract_ngrams("a an the is") == []


def test_compute_term_burst_bursting_topic():
    # 6 reviews with "broken wheel" from 4 different users
    texts = [
        "Broken wheel on second day",
        "The wheel broken completely",
        "Another broken wheel issue",
        "Terrible, broken wheel",
        "Broken wheel again",
        "Wheel broken while walking",
    ]
    uids = ["u1", "u2", "u3", "u4", "u1", "u2"]

    score, term = compute_term_burst(
        texts,
        uids,
        prior_texts=[],
        min_term_count=5,
        min_distinct_users=3,
    )
    assert score > 0.0
    assert term in ("broken", "wheel", "broken wheel")


def test_compute_term_burst_rejected_by_distinct_users():
    # 6 reviews with "battery exploded" but all from 1 single user (spam/single reporter)
    texts = ["Battery exploded", "Battery exploded again", "Battery exploded now", "Battery exploded", "Battery exploded", "Battery exploded"]
    uids = ["spammer", "spammer", "spammer", "spammer", "spammer", "spammer"]

    score, term = compute_term_burst(
        texts,
        uids,
        prior_texts=[],
        min_term_count=5,
        min_distinct_users=3,
    )
    # Should not trigger burst because distinct users < 3
    assert score == 0.0
    assert term is None


def test_compute_term_burst_rejected_by_min_count():
    # Only 2 reviews with "fire hazard"
    texts = ["Fire hazard on cord", "Huge fire hazard"]
    uids = ["u1", "u2"]

    score, term = compute_term_burst(
        texts,
        uids,
        prior_texts=[],
        min_term_count=5,
        min_distinct_users=2,
    )
    assert score == 0.0
    assert term is None


def test_compute_term_burst_with_prior_baseline():
    # Current window has 6 occurrences
    curr_texts = ["defect issue"] * 6
    curr_uids = [f"u{i}" for i in range(6)]

    # Case A: Prior window had 0 occurrences -> high surge
    score_new, _ = compute_term_burst(
        curr_texts, curr_uids, prior_texts=[], min_term_count=5, min_distinct_users=3
    )

    # Case B: Prior window already had 10 occurrences -> normal or declining frequency
    prior_texts = ["defect issue"] * 10
    score_declining, _ = compute_term_burst(
        curr_texts, curr_uids, prior_texts=prior_texts, min_term_count=5, min_distinct_users=3
    )

    assert score_new > score_declining
    assert score_declining == 0.0  # (6 - 10) / sqrt(11) < 0 -> clamped to 0.0


def test_compute_topic_window_features():
    dates = [date(2023, 1, 1 + i) for i in range(10)]
    reviews_df = pl.DataFrame(
        {
            "date": [date(2023, 1, 5), date(2023, 1, 6), date(2023, 1, 7)],
            "user_id": ["u1", "u2", "u3"],
            "text_norm": ["toxic odor inside", "bad toxic odor", "horrible toxic odor"],
        }
    )

    rows = compute_topic_window_features(
        reviews_df,
        dates,
        sku="SKU_01",
        category="Baby_Products",
        window_days=7,
        min_term_count=3,
        min_distinct_users=3,
    )

    # 10 days with window_days=7 -> 4 rolling windows
    assert len(rows) == 4
    for r in rows:
        assert r["sku"] == "SKU_01"
        assert r["indicator_id"] == "I7"
        assert r["feature"] == "term_burst_score"
        assert isinstance(r["window_end"], date)
