"""Kiểm thử cho Stage 3: Prompts SCCT, Anti-Hallucination Guardrails & Fallback Template."""

import json
from datetime import date
import pytest

from brandsentinel.llm.evidence import build_evidence_pack
from brandsentinel.llm.fallback import generate_fallback_recommendation
from brandsentinel.llm.prompts import (
    build_scct_system_prompt,
    build_scct_user_prompt,
    format_messages_for_llm,
)
from brandsentinel.llm.schemas import (
    ActionItem,
    DepartmentActions,
    EvidencePack,
    IndicatorEvidence,
    PriorityLevel,
    ProbableCause,
    RecommendationOutput,
    ReviewEvidence,
    SCCTCrisisType,
)
from brandsentinel.llm.validate import (
    extract_json_from_text,
    parse_and_validate_recommendation,
    validate_citations,
)


@pytest.fixture
def sample_evidence_pack() -> EvidencePack:
    return EvidencePack(
        sku="B0MOCK0000",
        category="Baby_Products",
        window_end="2023-03-15",
        risk_score=0.85,
        risk_level="HIGH",
        groups_active=["volume", "rating", "content"],
        persist=2,
        verify_flag=False,
        triggered_indicators=[
            IndicatorEvidence(indicator_id="I1", name="Volume Spike", triggered=True),
            IndicatorEvidence(indicator_id="I8", name="Safety Risk", triggered=True),
        ],
        reviews=[
            ReviewEvidence(
                review_id="rev_valid_1",
                rating=1,
                text="The charger overheated and melted the plastic!",
                date="2023-03-14",
                risk_hit=True,
                risk_terms=["overheat", "melt"],
            ),
            ReviewEvidence(
                review_id="rev_valid_2",
                rating=2,
                text="Dangerous sparks when plugging in.",
                date="2023-03-15",
                risk_hit=True,
                risk_terms=["spark"],
            ),
        ],
    )


def test_prompts_structure(sample_evidence_pack):
    """Kiểm tra cấu trúc và nội dung prompt SCCT."""
    sys_prompt = build_scct_system_prompt()
    assert "SCCT" in sys_prompt
    assert "Victim" in sys_prompt
    assert "Accidental" in sys_prompt
    assert "Preventable" in sys_prompt
    assert "QA_KyThuat" in sys_prompt
    assert "VanHanh" in sys_prompt
    assert "CSKH" in sys_prompt
    assert "TruyenThong" in sys_prompt
    assert "ANTI-HALLUCINATION" in sys_prompt

    user_prompt = build_scct_user_prompt(sample_evidence_pack)
    assert sample_evidence_pack.sku in user_prompt
    assert "rev_valid_1" in user_prompt
    assert "rev_valid_2" in user_prompt

    messages = format_messages_for_llm(sample_evidence_pack)
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"


def test_validate_citations_success(sample_evidence_pack):
    """Trích dẫn các review_id có trong pack sẽ pass kiểm tra."""
    rec = RecommendationOutput(
        sku=sample_evidence_pack.sku,
        category=sample_evidence_pack.category,
        window_end=sample_evidence_pack.window_end,
        risk_score=sample_evidence_pack.risk_score,
        summary="Củ sạc bị chập cháy nghiêm trọng.",
        scct_crisis_type=SCCTCrisisType.PREVENTABLE,
        probable_causes=[
            ProbableCause(
                cause="Lỗi biến áp lô sạc",
                evidence_review_ids=["rev_valid_1", "rev_valid_2"],
                confidence=0.9,
            )
        ],
        actions=DepartmentActions(
            QA_KyThuat=[ActionItem(priority=PriorityLevel.HIGH, action="Kiểm tra lô sạc", timeline="12h")],
            VanHanh=[ActionItem(priority=PriorityLevel.HIGH, action="Cách ly kho", timeline="24h")],
            CSKH=[ActionItem(priority=PriorityLevel.HIGH, action="Hoàn tiền khách", timeline="24h")],
            TruyenThong=[ActionItem(priority=PriorityLevel.HIGH, action="Thông cáo báo chí", timeline="24h")],
        ),
    )

    errors = validate_citations(rec, sample_evidence_pack)
    assert errors == []


def test_validate_citations_hallucination_caught(sample_evidence_pack):
    """Nếu LLM tự bịa ra review_id lạ, validator phải bắt được lỗi ảo giác."""
    rec = RecommendationOutput(
        sku=sample_evidence_pack.sku,
        category=sample_evidence_pack.category,
        window_end=sample_evidence_pack.window_end,
        risk_score=sample_evidence_pack.risk_score,
        summary="Củ sạc bị chập cháy.",
        scct_crisis_type=SCCTCrisisType.PREVENTABLE,
        probable_causes=[
            ProbableCause(
                cause="Lỗi biến áp",
                evidence_review_ids=["rev_valid_1", "FAKE_REVIEW_99999"],  # ID không có trong pack!
                confidence=0.9,
            )
        ],
        actions=DepartmentActions(
            QA_KyThuat=[ActionItem(priority=PriorityLevel.HIGH, action="Kiểm tra", timeline="12h")],
            VanHanh=[ActionItem(priority=PriorityLevel.HIGH, action="Cách ly", timeline="24h")],
            CSKH=[ActionItem(priority=PriorityLevel.HIGH, action="Hoàn tiền", timeline="24h")],
            TruyenThong=[ActionItem(priority=PriorityLevel.HIGH, action="Thông cáo", timeline="24h")],
        ),
    )

    errors = validate_citations(rec, sample_evidence_pack)
    assert len(errors) == 1
    assert "FAKE_REVIEW_99999" in errors[0]
    assert "Ảo giác trích dẫn" in errors[0]


def test_parse_and_validate_markdown_json(sample_evidence_pack):
    """Kiểm tra bóc tách JSON nằm trong markdown block ```json ... ```."""
    mock_llm_markdown = f"""
Chắc chắn rồi, đây là báo cáo phân tích theo SCCT:
```json
{{
  "sku": "{sample_evidence_pack.sku}",
  "category": "{sample_evidence_pack.category}",
  "window_end": "{sample_evidence_pack.window_end}",
  "risk_score": 0.85,
  "summary": "Nhiều phản ánh nghiêm trọng về việc củ sạc quá nhiệt và phát tia lửa.",
  "scct_crisis_type": "Preventable",
  "probable_causes": [
    {{
      "cause": "Lỗi cách điện gây ra hiện tượng đánh lửa và quá nhiệt",
      "evidence_review_ids": ["rev_valid_1"],
      "confidence": 0.95
    }}
  ],
  "actions": {{
    "QA_KyThuat": [{{"priority": "HIGH", "action": "Kiểm định linh kiện cách điện", "timeline": "12h"}}],
    "VanHanh": [{{"priority": "HIGH", "action": "Tạm dừng phân phối lô tuần 34", "timeline": "24h"}}],
    "CSKH": [{{"priority": "HIGH", "action": "Liên hệ bồi thường", "timeline": "24h"}}],
    "TruyenThong": [{{"priority": "HIGH", "action": "Chuẩn bị kịch bản phản hồi", "timeline": "24h"}}]
  }}
}}
```
Hy vọng phản hồi này giúp ích!
"""
    rec, errors = parse_and_validate_recommendation(mock_llm_markdown, sample_evidence_pack)
    assert errors == []
    assert rec is not None
    assert rec.scct_crisis_type == SCCTCrisisType.PREVENTABLE
    assert rec.probable_causes[0].evidence_review_ids == ["rev_valid_1"]


def test_fallback_recommendations(sample_evidence_pack):
    """Kiểm tra template fallback tự động thích ứng với kịch bản I8 và verify_flag."""
    # Ca 1: Có I8 -> SCCT: Preventable
    fb_preventable = generate_fallback_recommendation(sample_evidence_pack, reason="OpenAI timeout")
    assert fb_preventable.is_fallback is True
    assert fb_preventable.scct_crisis_type == SCCTCrisisType.PREVENTABLE
    assert "CẢNH BÁO AN TOÀN" in fb_preventable.summary
    assert fb_preventable.actions.QA_KyThuat[0].priority == PriorityLevel.HIGH

    # Kiểm tra fallback luôn tự pass validation
    _, errors = parse_and_validate_recommendation(fb_preventable.model_dump(), sample_evidence_pack)
    assert errors == []

    # Ca 2: verify_flag = True -> SCCT: Victim
    pack_victim = sample_evidence_pack.model_copy(
        update={"verify_flag": True, "triggered_indicators": [IndicatorEvidence(indicator_id="I11", name="Bombing", triggered=True)]}
    )
    fb_victim = generate_fallback_recommendation(pack_victim, reason="Rate limit")
    assert fb_victim.scct_crisis_type == SCCTCrisisType.VICTIM
    assert "review bombing" in fb_victim.summary.lower()

    # Ca 3: Lỗi thông thường (không I8, không cờ) -> SCCT: Accidental
    pack_accidental = sample_evidence_pack.model_copy(
        update={"verify_flag": False, "triggered_indicators": [IndicatorEvidence(indicator_id="I4", name="Low Star", triggered=True)]}
    )
    fb_accidental = generate_fallback_recommendation(pack_accidental, reason="Service error")
    assert fb_accidental.scct_crisis_type == SCCTCrisisType.ACCIDENTAL
