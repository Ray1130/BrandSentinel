"""Pydantic schemas cho Module LLM (M5): Evidence Pack, SCCT Recommendation và LLM Configuration.

Tuân thủ nghiêm ngặt docs/data_contract.md (Mục 3.7):
- Input: EvidencePack (gói bằng chứng chứa SKU, điểm rủi ro, chỉ báo kích hoạt, top review trích dẫn).
- Output: RecommendationOutput (tóm tắt, nguyên nhân gốc rễ, phân loại khủng hoảng SCCT, 4 phòng ban).
- Config: LLMConfig (nạp từ configs/llm.yaml, kiểm soát ngân sách, model, cache).
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class _BaseSchema(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


# ==============================================================================
# 1. Cấu hình LLM (Mapping từ configs/llm.yaml)
# ==============================================================================


class LLMConfig(_BaseSchema):
    """Cấu hình hoàn chỉnh cho module LLM, tương thích trực tiếp với configs/llm.yaml."""

    provider: Literal["openai", "groq", "mock", "custom"] = "openai"
    model: str = "gpt-4o-mini"
    base_url: str = "https://api.openai.com/v1"
    api_key_env: str = "OPENAI_API_KEY"
    monthly_budget_usd: float = Field(default=5.0, gt=0.0, description="Hạn mức ngân sách tối đa (USD)")
    cache_enabled: bool = True
    cache_key: Literal["sha256_canonical_evidence"] = "sha256_canonical_evidence"
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1500, ge=100, le=4096)
    timeout_seconds: int = Field(default=30, ge=5, le=120)
    max_retries: int = Field(default=2, ge=0, le=5)
    max_top_reviews: int = Field(default=5, ge=1, le=50)
    mask_pii: bool = True

    # Chi phí token ước tính (gpt-4o-mini: $0.15/1M input, $0.60/1M output)
    cost_per_1k_input_tokens: float = Field(default=0.00015, ge=0.0)
    cost_per_1k_output_tokens: float = Field(default=0.00060, ge=0.0)

    @classmethod
    def from_yaml(cls, path: Path | str) -> LLMConfig:
        """Đọc file YAML cấu hình."""
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"Không tìm thấy file cấu hình LLM: {p}")
        with p.open(encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ValueError(f"File {p} không chứa định dạng YAML mapping hợp lệ")
        return cls.model_validate(data)



# ==============================================================================
# 2. Input: Gói bằng chứng (Evidence Pack)
# ==============================================================================


class ReviewEvidence(_BaseSchema):
    """Một review đại diện làm bằng chứng."""

    review_id: str = Field(description="Khóa chính review_id (16 ký tự hex sha1)")
    rating: int = Field(ge=1, le=5, description="Số sao 1-5")
    text: str = Field(description="Nội dung đã chuẩn hóa hoặc làm sạch (đã che PII nếu bật)")
    date: str = Field(description="Ngày đăng (YYYY-MM-DD)")
    verified_purchase: bool | None = None
    risk_hit: bool = False
    risk_terms: list[str] = Field(default_factory=list)
    aspect: str | None = None


class IndicatorEvidence(_BaseSchema):
    """Chỉ báo rủi ro đã bị kích hoạt."""

    indicator_id: str = Field(description="Mã chỉ báo I1..I11")
    name: str = Field(description="Tên mô tả chỉ báo")
    triggered: bool = True
    strength: float | None = Field(default=None, description="Độ mạnh thống kê (robust z-score)")


class EvidencePack(_BaseSchema):
    """Gói bằng chứng hoàn chỉnh cho một SKU đạt cảnh báo HIGH."""

    sku: str = Field(description="Mã sản phẩm SKU (parent_asin)")
    category: str = Field(description="Ngành hàng sản phẩm")
    window_end: str = Field(description="Mốc ngày cuối cửa sổ phân tích (YYYY-MM-DD)")
    risk_score: float = Field(ge=0.0, le=1.0, description="RiskScore tổng hợp")
    risk_level: Literal["HIGH"] = "HIGH"
    groups_active: list[str] = Field(description="Danh sách nhóm tín hiệu active (volume, rating, content)")
    persist: int = Field(ge=0, description="Số cửa sổ liên tiếp duy trì rủi ro")
    verify_flag: bool = Field(default=False, description="Cờ nghi ngờ review bombing / bất thường")
    triggered_indicators: list[IndicatorEvidence] = Field(default_factory=list)
    reviews: list[ReviewEvidence] = Field(
        min_length=1, description="Danh sách review tiêu biểu làm căn cứ trích dẫn"
    )
    summary_stats: dict[str, Any] = Field(default_factory=dict)

    @property
    def review_ids_set(self) -> set[str]:
        """Tập hợp tất cả review_id hợp lệ có trong gói bằng chứng."""
        return {r.review_id for r in self.reviews}


# ==============================================================================
# 3. Output: Khuyến nghị hành động (SCCT Recommendations)
# ==============================================================================


class PriorityLevel(StrEnum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class SCCTCrisisType(StrEnum):
    """Phân loại cụm khủng hoảng theo Situational Crisis Communication Theory (Coombs, 2002).

    - VICTIM: Doanh nghiệp là nạn nhân (review bombing, tấn công ác ý, tin giả ngoài tầm kiểm soát).
    - ACCIDENTAL: Sự cố ngoài ý muốn (lỗi kỹ thuật ngẫu nhiên, sự cố kỹ thuật đối tác không lường trước).
    - PREVENTABLE: Có thể phòng ngừa (lỗi chất lượng dây chuyền, khuyết tật sản phẩm đã biết, lỗi chủ quan).
    """

    VICTIM = "Victim"
    ACCIDENTAL = "Accidental"
    PREVENTABLE = "Preventable"


class ActionItem(_BaseSchema):
    """Một khuyến nghị hành động cụ thể cho phòng ban."""

    priority: PriorityLevel = Field(description="Mức độ ưu tiên thực hiện")
    action: str = Field(min_length=5, description="Nội dung hành động cụ thể, rõ ràng")
    timeline: str = Field(min_length=2, description="Khung thời gian thực hiện (ví dụ: '24 giờ', '7 ngày')")


class DepartmentActions(_BaseSchema):
    """Khuyến nghị phân vai cho 4 phòng ban nòng cốt (Thuyết minh trang 21-22)."""

    QA_KyThuat: list[ActionItem] = Field(
        min_length=1, description="Bộ phận Kiểm soát chất lượng & Kỹ thuật sản phẩm"
    )
    VanHanh: list[ActionItem] = Field(min_length=1, description="Bộ phận Vận hành & Quản lý chuỗi cung ứng")
    CSKH: list[ActionItem] = Field(min_length=1, description="Bộ phận Chăm sóc khách hàng")
    TruyenThong: list[ActionItem] = Field(min_length=1, description="Bộ phận Truyền thông & Quản trị thương hiệu")


class ProbableCause(_BaseSchema):
    """Giả thuyết nguyên nhân gốc rễ kèm bằng chứng trích dẫn."""

    cause: str = Field(min_length=5, description="Mô tả nguyên nhân nghi vấn")
    evidence_review_ids: list[str] = Field(
        min_length=1, description="Danh sách review_id BẮT BUỘC phải nằm trong gói bằng chứng"
    )
    confidence: float = Field(ge=0.0, le=1.0, description="Mức độ chắc chắn (0.0 đến 1.0)")

    @field_validator("evidence_review_ids")
    @classmethod
    def _strip_and_dedup(cls, v: list[str]) -> list[str]:
        cleaned = [item.strip() for item in v if item.strip()]
        if not cleaned:
            raise ValueError("Cần ít nhất một review_id hợp lệ làm bằng chứng")
        return list(dict.fromkeys(cleaned))


class RecommendationOutput(_BaseSchema):
    """Đầu ra JSON chuẩn lưu tại processed/recommendations.json (data_contract.md 3.7)."""

    sku: str = Field(description="Mã SKU tương ứng")
    category: str = Field(description="Ngành hàng")
    window_end: str = Field(description="Mốc thời gian phân tích (YYYY-MM-DD)")
    risk_score: float = Field(ge=0.0, le=1.0, description="RiskScore từ bảng alerts")
    summary: str = Field(min_length=10, description="Tóm tắt phản ánh nổi bật của khách hàng")
    scct_crisis_type: SCCTCrisisType = Field(description="Phân loại khủng hoảng theo SCCT")
    probable_causes: list[ProbableCause] = Field(
        min_length=1, description="Danh sách các nguyên nhân nghi vấn kèm trích dẫn"
    )
    actions: DepartmentActions = Field(description="Kế hoạch hành động chi tiết cho 4 phòng ban")
    generated_at: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
        description="Thời điểm sinh khuyến nghị (ISO format UTC)",
    )
    model_used: str = Field(default="gpt-4o-mini", description="Tên mô hình LLM hoặc Template sử dụng")
    is_fallback: bool = Field(default=False, description="True nếu sinh từ fallback template")
