"""Kiểm thử hợp đồng dữ liệu cho module LLM: schemas.py và configs/llm.yaml."""

from pathlib import Path
import pytest
from pydantic import ValidationError

from brandsentinel.llm.schemas import (
    ActionItem,
    DepartmentActions,
    EvidencePack,
    IndicatorEvidence,
    LLMConfig,
    PriorityLevel,
    ProbableCause,
    RecommendationOutput,
    ReviewEvidence,
    SCCTCrisisType,
)


def test_load_llm_yaml_config():
    """Kiểm tra đọc file cấu hình configs/llm.yaml."""
    cfg_path = Path(__file__).resolve().parents[2] / "configs" / "llm.yaml"
    cfg = LLMConfig.from_yaml(cfg_path)

    assert cfg.provider == "openai"
    assert cfg.model == "gpt-4o-mini"
    assert cfg.api_key_env == "OPENAI_API_KEY"
    assert cfg.monthly_budget_usd == 5.0
    assert cfg.cache_enabled is True
    assert cfg.cache_key == "sha256_canonical_evidence"
    assert cfg.cost_per_1k_input_tokens > 0
    assert cfg.cost_per_1k_output_tokens > 0



def test_evidence_pack_validation():
    """Kiểm tra tính hợp lệ của EvidencePack."""
    pack = EvidencePack(
        sku="B0MOCK0000",
        category="Baby_Products",
        window_end="2023-03-15",
        risk_score=0.85,
        risk_level="HIGH",
        groups_active=["volume", "rating", "content"],
        persist=3,
        verify_flag=False,
        triggered_indicators=[
            IndicatorEvidence(indicator_id="I1", name="Volume Spike", triggered=True, strength=3.2),
            IndicatorEvidence(indicator_id="I8", name="Safety Risk", triggered=True, strength=4.5),
        ],
        reviews=[
            ReviewEvidence(
                review_id="rev_001",
                rating=1,
                text="The charger overheated and melted!",
                date="2023-03-14",
                risk_hit=True,
                risk_terms=["overheat", "melt"],
            ),
            ReviewEvidence(
                review_id="rev_002",
                rating=2,
                text="Smelled smoke coming from the device.",
                date="2023-03-15",
                risk_hit=True,
                risk_terms=["smoke"],
            ),
        ],
    )

    assert pack.sku == "B0MOCK0000"
    assert pack.risk_score == 0.85
    assert pack.review_ids_set == {"rev_001", "rev_002"}
    assert len(pack.triggered_indicators) == 2


def test_evidence_pack_empty_reviews_rejected():
    """EvidencePack không được phép rỗng reviews."""
    with pytest.raises(ValidationError):
        EvidencePack(
            sku="B0MOCK0000",
            category="Baby_Products",
            window_end="2023-03-15",
            risk_score=0.85,
            risk_level="HIGH",
            groups_active=["volume"],
            persist=1,
            reviews=[],  # Vi phạm min_length=1
        )


def test_recommendation_output_valid():
    """Kiểm tra schema khuyến nghị đầy đủ 4 bộ phận và SCCT."""
    output = RecommendationOutput(
        sku="B0MOCK0000",
        category="Baby_Products",
        window_end="2023-03-15",
        risk_score=0.85,
        summary="Phát hiện nhiều phản hồi nghiêm trọng về việc củ sạc quá nhiệt và bốc khói.",
        scct_crisis_type=SCCTCrisisType.PREVENTABLE,
        probable_causes=[
            ProbableCause(
                cause="Lỗi bộ biến áp hoặc linh kiện cách điện của lô sạc tuần 34",
                evidence_review_ids=["rev_001", "rev_002"],
                confidence=0.92,
            )
        ],
        actions=DepartmentActions(
            QA_KyThuat=[
                ActionItem(
                    priority=PriorityLevel.HIGH,
                    action="Tạm dừng xuất kho lô sạc sản xuất tuần 34 và kiểm tra mạch bảo vệ nhiệt.",
                    timeline="Ngay lập tức (trong 12 giờ)",
                )
            ],
            VanHanh=[
                ActionItem(
                    priority=PriorityLevel.HIGH,
                    action="Kiểm tra kho lưu trữ và cách ly các kiện hàng cùng batch ID.",
                    timeline="24 giờ",
                )
            ],
            CSKH=[
                ActionItem(
                    priority=PriorityLevel.HIGH,
                    action="Chủ động liên hệ hoàn tiền 100% cho các khách hàng khiếu nại nhiệt độ.",
                    timeline="24-48 giờ",
                )
            ],
            TruyenThong=[
                ActionItem(
                    priority=PriorityLevel.MEDIUM,
                    action="Chuẩn bị sẵn thông cáo hướng dẫn sử dụng an toàn; kích hoạt kịch bản Preventable crisis.",
                    timeline="48 giờ",
                )
            ],
        ),
    )

    data_json = output.model_dump_json()
    assert "QA_KyThuat" in data_json
    assert "VanHanh" in data_json
    assert "CSKH" in data_json
    assert "TruyenThong" in data_json
    assert output.scct_crisis_type == SCCTCrisisType.PREVENTABLE


def test_recommendation_output_missing_department_rejected():
    """Không được thiếu bất kỳ phòng ban nào trong 4 phòng ban cốt lõi."""
    with pytest.raises(ValidationError):
        RecommendationOutput(
            sku="B0MOCK0000",
            category="Baby_Products",
            window_end="2023-03-15",
            risk_score=0.85,
            summary="Tóm tắt lỗi",
            scct_crisis_type=SCCTCrisisType.ACCIDENTAL,
            probable_causes=[
                ProbableCause(cause="Lỗi pin", evidence_review_ids=["rev_01"], confidence=0.8)
            ],
            actions=DepartmentActions(
                QA_KyThuat=[ActionItem(priority=PriorityLevel.HIGH, action="Kiểm tra", timeline="24h")],
                VanHanh=[ActionItem(priority=PriorityLevel.HIGH, action="Kiểm tra", timeline="24h")],
                CSKH=[],  # Rỗng -> vi phạm min_length=1
                TruyenThong=[ActionItem(priority=PriorityLevel.HIGH, action="Kiểm tra", timeline="24h")],
            ),
        )


def test_probable_cause_empty_reviews_rejected():
    """Mỗi giả thuyết nguyên nhân bắt buộc phải có review_id làm bằng chứng."""
    with pytest.raises(ValidationError):
        ProbableCause(
            cause="Lỗi sản phẩm chung chung",
            evidence_review_ids=[],  # Vi phạm
            confidence=0.5,
        )
