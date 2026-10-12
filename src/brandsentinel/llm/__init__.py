"""Module LLM (M5): Diễn giải rủi ro, phân tích khủng hoảng SCCT và đề xuất khuyến nghị."""

from brandsentinel.llm.client import (
    call_llm_with_guardrails,
    get_api_key_or_none,
    is_budget_exceeded,
    read_budget_state,
    record_token_usage,
    require_api_key,
)
from brandsentinel.llm.evidence import (
    INDICATOR_DESCRIPTIONS,
    build_evidence_pack,
    canonical_evidence_dict,
    extract_high_alerts_evidence_packs,
    mask_pii,
    select_representative_reviews,
)
from brandsentinel.llm.fallback import generate_fallback_recommendation
from brandsentinel.llm.prompts import (
    SYSTEM_PROMPT,
    build_scct_system_prompt,
    build_scct_user_prompt,
    format_messages_for_llm,
)
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
from brandsentinel.llm.validate import (
    extract_json_from_text,
    parse_and_validate_recommendation,
    validate_citations,
)

__all__ = [
    "ActionItem",
    "DepartmentActions",
    "EvidencePack",
    "IndicatorEvidence",
    "LLMConfig",
    "PriorityLevel",
    "ProbableCause",
    "RecommendationOutput",
    "ReviewEvidence",
    "SCCTCrisisType",
    "INDICATOR_DESCRIPTIONS",
    "build_evidence_pack",
    "canonical_evidence_dict",
    "extract_high_alerts_evidence_packs",
    "mask_pii",
    "select_representative_reviews",
    "SYSTEM_PROMPT",
    "build_scct_system_prompt",
    "build_scct_user_prompt",
    "format_messages_for_llm",
    "generate_fallback_recommendation",
    "extract_json_from_text",
    "parse_and_validate_recommendation",
    "validate_citations",
    "require_api_key",
    "get_api_key_or_none",
    "call_llm_with_guardrails",
    "record_token_usage",
    "read_budget_state",
    "is_budget_exceeded",
]
