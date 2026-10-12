"""Kiểm thử cho Stage 2 (Evidence Pack Generator) trong module LLM."""

from datetime import date
import polars as pl
import pytest

from brandsentinel.core.types import Level, Table
from brandsentinel.llm.cache import evidence_hash
from brandsentinel.llm.evidence import (
    build_evidence_pack,
    canonical_evidence_dict,
    extract_high_alerts_evidence_packs,
    mask_pii,
    select_representative_reviews,
)
from brandsentinel.testing.mock_data import make_all


def test_mask_pii():
    """Kiểm tra che email, số điện thoại và URL."""
    text = (
        "Please contact me at customer_service@shop.com or call 555-234-5678. "
        "Also check https://fake-review.com for details. The product caught fire!"
    )
    masked = mask_pii(text)
    assert "[EMAIL]" in masked
    assert "customer_service@shop.com" not in masked
    assert "[PHONE]" in masked
    assert "555-234-5678" not in masked
    assert "[URL]" in masked
    assert "caught fire!" in masked


def test_select_representative_reviews_priority():
    """Kiểm tra độ ưu tiên chọn review: risk_hit cao nhất, rồi đến rating thấp, loại spam."""
    reviews_df = pl.DataFrame({
        "review_id": ["r1", "r2", "r3", "r4", "r5"],
        "rating": [5, 1, 1, 2, 1],
        "text_raw": [
            "Great product, works perfectly for our baby.",
            "Spam bot link buy now",
            "Battery overheated and started smoking, very dangerous!",
            "Terrible build quality, plastic cracked after two days.",
            "Battery overheated and started smoking, very dangerous!",  # Trùng nội dung với r3
        ],
        "text_norm": [
            "great product works perfectly for our baby",
            "spam bot link buy now",
            "battery overheated and started smoking very dangerous",
            "terrible build quality plastic cracked after two days",
            "battery overheated and started smoking very dangerous",
        ],
        "date": [date(2023, 3, 10), date(2023, 3, 11), date(2023, 3, 12), date(2023, 3, 12), date(2023, 3, 12)],
        "verified_purchase": [True, False, True, True, True],
        "is_spam": [False, True, False, False, False],
        "risk_hit": [False, False, True, False, True],
        "risk_terms": [[], [], ["overheat", "smoke"], [], ["overheat", "smoke"]],
        "aspect": ["quality", None, "safety", "quality", "safety"],
    })

    selected = select_representative_reviews(reviews_df, max_reviews=3, mask_pii_enabled=True)

    # R2 bị loại vì is_spam = True
    assert not any(r.review_id == "r2" for r in selected)

    # R3 có risk_hit = True nên phải đứng đầu
    assert selected[0].review_id == "r3"
    assert selected[0].risk_hit is True
    assert "overheat" in selected[0].risk_terms

    # R5 bị loại trùng lặp nội dung với r3
    assert not any(r.review_id == "r5" for r in selected)

    # R4 (1-2 sao) được ưu tiên hơn R1 (5 sao)
    review_ids = [r.review_id for r in selected]
    assert "r4" in review_ids
    assert len(selected) <= 3


def test_build_evidence_pack_single():
    """Kiểm tra đóng gói một EvidencePack từ alert record."""
    alert_rec = {
        "sku": "B0MOCK0000",
        "category": "Baby_Products",
        "window_end": date(2023, 3, 15),
        "score": 0.85,
        "level": "HIGH",
        "groups_active": ["volume", "rating", "content"],
        "persist": 2,
        "verify_flag": False,
        "triggered_ids": ["I1", "I8"],
    }

    clean_df = pl.DataFrame({
        "sku": ["B0MOCK0000", "B0MOCK0000"],
        "review_id": ["rev_1", "rev_2"],
        "date": [date(2023, 3, 12), date(2023, 3, 14)],
        "rating": [1, 2],
        "text_raw": ["Melting plug", "Device broke and smelled bad"],
        "text_norm": ["melting plug", "device broke and smelled bad"],
        "verified_purchase": [True, True],
        "is_spam": [False, False],
    })

    pack = build_evidence_pack(
        alert_record=alert_rec,
        clean_reviews_df=clean_df,
        window_days=7,
        max_reviews=5,
    )

    assert pack.sku == "B0MOCK0000"
    assert pack.risk_level == "HIGH"
    assert pack.risk_score == 0.85
    assert len(pack.triggered_indicators) == 2
    assert pack.triggered_indicators[0].indicator_id == "I1"
    assert pack.triggered_indicators[1].indicator_id == "I8"
    assert len(pack.reviews) == 2
    assert pack.review_ids_set == {"rev_1", "rev_2"}


def test_extract_high_alerts_evidence_packs_on_mock_data():
    """Kiểm tra chạy trên tập dữ liệu mock hoàn chỉnh make_all()."""
    tables = make_all(seed=42, n_skus=6, days=120)
    alerts_df = tables[Table.ALERTS]
    clean_df = tables[Table.CLEAN_REVIEWS]
    nlp_df = tables[Table.NLP_FEATURES]

    # Kiểm tra chỉ có alert HIGH được chọn
    high_count = alerts_df.filter(pl.col("level") == Level.HIGH.value).height
    assert high_count > 0, "Dữ liệu mock cần có ít nhất 1 alert HIGH"

    packs = extract_high_alerts_evidence_packs(
        alerts_df=alerts_df,
        clean_reviews_df=clean_df,
        nlp_features_df=nlp_df,
        min_level="HIGH",
    )

    assert len(packs) == high_count
    for pack in packs:
        assert pack.risk_level == "HIGH"
        assert len(pack.reviews) >= 1
        # Mọi review phải có review_id hợp lệ
        for r in pack.reviews:
            assert len(r.review_id) > 0
            assert 1 <= r.rating <= 5


def test_canonical_evidence_dict_and_caching():
    """Kiểm tra EvidencePack có thể chuyển sang canonical dict và hash SHA-256 nhất quán."""
    alert_rec = {
        "sku": "B0TEST",
        "category": "Electronics",
        "window_end": "2023-04-01",
        "score": 0.7619,
        "level": "HIGH",
        "groups_active": ["rating", "content"],
        "persist": 1,
        "verify_flag": False,
        "triggered_ids": ["I4", "I8"],
    }
    clean_df = pl.DataFrame({
        "sku": ["B0TEST"],
        "review_id": ["rev_test_1"],
        "date": [date(2023, 4, 1)],
        "rating": [1],
        "text_raw": ["Battery exploded"],
        "text_norm": ["battery exploded"],
        "verified_purchase": [True],
        "is_spam": [False],
    })

    pack = build_evidence_pack(alert_rec, clean_df)
    c_dict = canonical_evidence_dict(pack)

    h1 = evidence_hash(c_dict)
    h2 = evidence_hash(c_dict)

    assert isinstance(h1, str)
    assert len(h1) == 64  # SHA-256 hexdigest
    assert h1 == h2
