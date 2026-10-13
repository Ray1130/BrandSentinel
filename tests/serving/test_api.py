"""Tests for FastAPI serving layer (brandsentinel.serving.api)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from brandsentinel.llm.schemas import (
    EvidencePack,
    IndicatorEvidence,
    RecommendationOutput,
    ReviewEvidence,
)
from brandsentinel.serving.api import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def sample_evidence_pack() -> EvidencePack:
    return EvidencePack(
        sku="B00SAMPLE01",
        category="Home_and_Kitchen",
        window_end="2023-08-30",
        risk_score=0.88,
        risk_level="HIGH",
        groups_active=["volume", "rating", "content"],
        persist=3,
        verify_flag=False,
        triggered_indicators=[
            IndicatorEvidence(
                indicator_id="I1",
                name="Volume spike",
                triggered=True,
                strength=3.2,
            ),
            IndicatorEvidence(
                indicator_id="I4",
                name="Low star ratio increase",
                triggered=True,
                strength=2.8,
            ),
            IndicatorEvidence(
                indicator_id="I8",
                name="Safety risk terms detected",
                triggered=True,
                strength=4.1,
            ),
        ],
        reviews=[
            ReviewEvidence(
                review_id="rev_smoke_001",
                rating=1,
                text="The blender started smoking and emitted burning smell after 2 minutes!",
                date="2023-08-28",
                verified_purchase=True,
                risk_hit=True,
                risk_terms=["smoking", "burning"],
                aspect="safety",
            ),
            ReviewEvidence(
                review_id="rev_spark_002",
                rating=1,
                text="Sparked from the base when turned on. Fire hazard!",
                date="2023-08-29",
                verified_purchase=True,
                risk_hit=True,
                risk_terms=["sparked", "fire hazard"],
                aspect="safety",
            ),
            ReviewEvidence(
                review_id="rev_break_003",
                rating=2,
                text="Plastic blade gear cracked on first use.",
                date="2023-08-29",
                verified_purchase=True,
                risk_hit=False,
                risk_terms=[],
                aspect="quality",
            ),
        ],
        summary_stats={"total_reviews_window": 35, "avg_rating_window": 1.4},
    )


def test_health_check(client: TestClient) -> None:
    """Kiểm tra health check endpoints."""
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert data["service"] == "BrandSentinel"

    res_v1 = client.get("/api/v1/health")
    assert res_v1.status_code == 200
    assert res_v1.json()["status"] == "ok"


def test_analyze_risk_force_fallback(
    client: TestClient, sample_evidence_pack: EvidencePack
) -> None:
    """Kiểm tra POST /api/v1/analyze-risk với force_fallback=True (offline safe)."""
    payload = sample_evidence_pack.model_dump()
    res = client.post("/api/v1/analyze-risk?force_fallback=true", json=payload)

    assert res.status_code == 200
    data = res.json()

    # Xác thực Pydantic schema
    rec = RecommendationOutput.model_validate(data)
    assert rec.sku == "B00SAMPLE01"
    assert rec.category == "Home_and_Kitchen"
    assert rec.is_fallback is True
    assert rec.risk_score == pytest.approx(0.88)

    # Đảm bảo phân công đủ 4 phòng ban
    assert len(rec.actions.QA_KyThuat) >= 1
    assert len(rec.actions.VanHanh) >= 1
    assert len(rec.actions.CSKH) >= 1
    assert len(rec.actions.TruyenThong) >= 1

    # Kiểm định trích dẫn: review_id trong probable_causes phải thuộc sample_evidence_pack
    valid_ids = sample_evidence_pack.review_ids_set
    assert len(rec.probable_causes) >= 1
    for cause in rec.probable_causes:
        assert len(cause.evidence_review_ids) >= 1
        for rid in cause.evidence_review_ids:
            assert rid in valid_ids, f"Review ID {rid} không thuộc EvidencePack!"


def test_analyze_risk_invalid_payload(client: TestClient) -> None:
    """Kiểm tra validation khi thiếu trường bắt buộc."""
    bad_payload = {"sku": "B001", "risk_score": 0.5}  # Thiếu nhiều trường
    res = client.post("/api/v1/analyze-risk", json=bad_payload)
    assert res.status_code == 422


def test_get_alerts_not_found(client: TestClient) -> None:
    """Kiểm tra GET /api/v1/alerts khi chưa có dữ liệu category."""
    res = client.get("/api/v1/alerts?category=CategoryKhongTonTai999")
    assert res.status_code == 404
    assert "Chưa có dữ liệu alerts" in res.json()["detail"]


def test_get_recommendations_not_found(client: TestClient) -> None:
    """Kiểm tra GET /api/v1/recommendations khi chưa có file recommendations.json."""
    res = client.get("/api/v1/recommendations?category=CategoryKhongTonTai999")
    assert res.status_code == 404
    assert "Chưa có recommendations.json" in res.json()["detail"]
