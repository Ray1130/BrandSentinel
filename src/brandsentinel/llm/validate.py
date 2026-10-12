"""Cơ chế kiểm định và chống ảo giác (Anti-Hallucination Guardrails) cho Module LLM (M5).

Tuân thủ nghiêm ngặt Thuyết minh dự án AISC'26:
- Mọi review_id được trích dẫn trong probable_causes BẮT BUỘC phải tồn tại trong gói bằng chứng đầu vào.
- LLM không được tự ý sửa mã SKU hoặc đổi cấp độ rủi ro.
- Kiểm tra tính đầy đủ của 4 phòng ban và định dạng Pydantic.
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import ValidationError

from brandsentinel.llm.schemas import EvidencePack, RecommendationOutput

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```")


def extract_json_from_text(raw_text: str) -> str:
    """Bóc tách chuỗi JSON sạch nếu LLM trả về trong khối markdown ```json ... ```."""
    stripped = raw_text.strip()
    match = _JSON_BLOCK_RE.search(stripped)
    if match:
        return match.group(1).strip()
    return stripped


def validate_citations(
    recommendation: RecommendationOutput,
    evidence_pack: EvidencePack,
) -> list[str]:
    """Kiểm tra tính toàn vẹn của bằng chứng: Mọi review_id trích dẫn phải có trong EvidencePack."""
    errors: list[str] = []
    valid_ids = evidence_pack.review_ids_set

    for idx, cause in enumerate(recommendation.probable_causes):
        for rid in cause.evidence_review_ids:
            if rid not in valid_ids:
                errors.append(
                    f"Ảo giác trích dẫn: review_id '{rid}' trong nguyên nhân #{idx+1} "
                    f"('{cause.cause[:40]}...') không tồn tại trong Gói bằng chứng."
                )

    return errors


def parse_and_validate_recommendation(
    raw_payload: str | dict[str, Any],
    evidence_pack: EvidencePack,
) -> tuple[RecommendationOutput | None, list[str]]:
    """Phân tích cú pháp và kiểm định toàn diện phản hồi từ LLM.

    Trả về:
        (RecommendationOutput, []) nếu hợp lệ 100%.
        (None, [danh sách lỗi]) nếu phát hiện vi phạm schema hoặc ảo giác trích dẫn.
    """
    errors: list[str] = []

    # 1. Parse JSON
    if isinstance(raw_payload, str):
        json_str = extract_json_from_text(raw_payload)
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            return None, [f"Lỗi cú pháp JSON không thể giải mã: {e}"]
    elif isinstance(raw_payload, dict):
        data = raw_payload
    else:
        return None, ["Định dạng đầu vào không hợp lệ (cần chuỗi JSON hoặc dictionary)."]

    # 2. Bổ sung các trường siêu dữ liệu nếu LLM bỏ quên
    if "sku" not in data or not data["sku"]:
        data["sku"] = evidence_pack.sku
    elif data["sku"] != evidence_pack.sku:
        errors.append(f"SKU không khớp: LLM trả về '{data['sku']}' thay vì '{evidence_pack.sku}'.")

    if "category" not in data or not data["category"]:
        data["category"] = evidence_pack.category

    if "window_end" not in data or not data["window_end"]:
        data["window_end"] = evidence_pack.window_end

    if "risk_score" not in data or data["risk_score"] is None:
        data["risk_score"] = evidence_pack.risk_score

    # 3. Validate Pydantic Schema
    try:
        recommendation = RecommendationOutput.model_validate(data)
    except ValidationError as e:
        for err in e.errors():
            loc = " -> ".join(str(x) for x in err["loc"])
            errors.append(f"Lỗi Schema tại '{loc}': {err['msg']}")
        return None, errors

    # 4. Anti-Hallucination: Validate review_id citations
    citation_errors = validate_citations(recommendation, evidence_pack)
    if citation_errors:
        errors.extend(citation_errors)
        return None, errors

    return recommendation, []
