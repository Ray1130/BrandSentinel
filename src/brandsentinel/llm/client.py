"""Client kết nối LLM với cơ chế Budget Tracking, Caching và Guardrails.

Tuân thủ:
1. Đọc API key từ biến môi trường mà không ghi log hay lưu trữ (require_api_key).
2. Bộ nhớ đệm Cache SHA-256 (canonical evidence hash) chống gọi trùng lặp.
3. Bộ theo dõi ngân sách (Budget Tracker / Circuit Breaker) chặn ngắt khi chạm trần 5.0 USD.
4. Tự động kiểm định và fallback sang Rule-based template khi có sự cố.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from brandsentinel.core.config import Config, LlmConfig, get_config
from brandsentinel.llm.cache import (
    evidence_hash,
    read_cached_recommendation,
    write_cached_recommendation,
)
from brandsentinel.llm.evidence import canonical_evidence_dict
from brandsentinel.llm.fallback import generate_fallback_recommendation
from brandsentinel.llm.prompts import format_messages_for_llm
from brandsentinel.llm.schemas import EvidencePack, RecommendationOutput
from brandsentinel.llm.validate import parse_and_validate_recommendation

log = logging.getLogger(__name__)

OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"


# ==============================================================================
# 1. Truy xuất thông tin xác thực (Credential Access)
# ==============================================================================


def require_api_key(config: LlmConfig) -> str:
    """Đọc secret từ biến môi trường. Ném lỗi nếu chưa cấu hình."""
    key = os.environ.get(config.api_key_env)
    if not key:
        raise RuntimeError(
            f"LLM API key is not set; provide it via the {config.api_key_env} environment variable"
        )
    return key


def get_api_key_or_none(config: LlmConfig) -> str | None:
    """Trả về API key nếu có, hoặc None nếu chưa thiết lập."""
    return os.environ.get(config.api_key_env)


# ==============================================================================
# 2. Bộ theo dõi ngân sách (Budget Tracker & Circuit Breaker)
# ==============================================================================


def get_budget_tracker_path(cfg: Config) -> Path:
    """Đường dẫn lưu trữ trạng thái tiêu dùng token và chi phí thực tế."""
    return cfg.path("data_processed") / "recommendations" / "budget_state.json"


def read_budget_state(cfg: Config) -> dict[str, Any]:
    """Đọc dữ liệu chi phí đã sử dụng."""
    path = get_budget_tracker_path(cfg)
    if not path.is_file():
        return {
            "total_cost_usd": 0.0,
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_calls": 0,
            "updated_at": datetime.now(UTC).isoformat(),
        }
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        log.warning("Không thể đọc budget_state.json: %s. Khởi tạo lại.", e)
        return {
            "total_cost_usd": 0.0,
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_calls": 0,
            "updated_at": datetime.now(UTC).isoformat(),
        }


def record_token_usage(
    cfg: Config,
    prompt_tokens: int,
    completion_tokens: int,
) -> float:
    """Ghi nhận token đã dùng, tính chi phí thực tế theo USD và cập nhật budget_state."""
    # Định mức giá gpt-4o-mini mặc định: $0.15 / 1M input, $0.60 / 1M output
    cost_in = (prompt_tokens / 1000.0) * 0.00015
    cost_out = (completion_tokens / 1000.0) * 0.00060
    call_cost = cost_in + cost_out

    state = read_budget_state(cfg)
    state["total_prompt_tokens"] = int(state.get("total_prompt_tokens", 0) + prompt_tokens)
    state["total_completion_tokens"] = int(state.get("total_completion_tokens", 0) + completion_tokens)
    state["total_cost_usd"] = float(round(state.get("total_cost_usd", 0.0) + call_cost, 6))
    state["total_calls"] = int(state.get("total_calls", 0) + 1)
    state["updated_at"] = datetime.now(UTC).isoformat()

    path = get_budget_tracker_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    return state["total_cost_usd"]


def is_budget_exceeded(cfg: Config, estimated_next_cost: float = 0.002) -> bool:
    """Kiểm tra xem chi phí đã vượt ngưỡng ngân sách (5.0 USD) hay chưa."""
    state = read_budget_state(cfg)
    current_spent = state.get("total_cost_usd", 0.0)
    limit = cfg.llm.monthly_budget_usd
    return (current_spent + estimated_next_cost) > limit


# ==============================================================================
# 3. Điều phối gọi LLM với Guardrails & Cache
# ==============================================================================


def call_llm_with_guardrails(
    pack: EvidencePack,
    cfg: Config | None = None,
    *,
    api_key: str | None = None,
    force_fallback: bool = False,
    http_client: httpx.Client | None = None,
) -> RecommendationOutput:
    """Hàm trung tâm gọi LLM sinh khuyến nghị cho một EvidencePack.

    Quy trình:
    1. Kiểm tra Cache SHA-256: Nếu có sẵn kết quả -> trả về ngay (0 chi phí).
    2. Nếu ép buộc fallback -> trả về Fallback template.
    3. Kiểm tra Ngân sách (Circuit Breaker): Nếu vượt 5 USD -> kích hoạt Fallback.
    4. Kiểm tra API Key: Nếu chưa có key trong môi trường -> kích hoạt Fallback an toàn.
    5. Gọi OpenAI API với cơ chế Retry tối đa 2 lần nếu phản hồi bị lỗi schema / trích dẫn.
    6. Lưu kết quả vào Cache và trả về RecommendationOutput.
    """
    app_cfg = cfg or get_config()
    canonical_dict = canonical_evidence_dict(pack)

    # 1. Kiểm tra cache
    if app_cfg.llm.cache_enabled:
        cached = read_cached_recommendation(app_cfg, canonical_dict)
        if cached is not None:
            log.info("Cache hit cho SKU %s (hash: %s)", pack.sku, evidence_hash(canonical_dict))
            try:
                return RecommendationOutput.model_validate(cached)
            except Exception as e:
                log.warning("Cache data không hợp lệ cho SKU %s: %s. Chạy lại.", pack.sku, e)

    # 2. Kiểm tra cờ ép buộc fallback (dùng cho offline testing)
    if force_fallback:
        log.info("Force fallback được kích hoạt cho SKU %s", pack.sku)
        rec = generate_fallback_recommendation(pack, reason="Forced rule-based mode")
        if app_cfg.llm.cache_enabled:
            write_cached_recommendation(app_cfg, canonical_dict, rec.model_dump())
        return rec

    # 3. Circuit breaker: Kiểm tra ngân sách
    if is_budget_exceeded(app_cfg):
        spent = read_budget_state(app_cfg).get("total_cost_usd", 0.0)
        log.warning(
            "Ngân sách LLM đã chạm hạn mức: đã tiêu $%.4f / $%.2f. Kích hoạt fallback cho SKU %s.",
            spent,
            app_cfg.llm.monthly_budget_usd,
            pack.sku,
        )
        rec = generate_fallback_recommendation(
            pack,
            reason=f"Ngân sách tháng ({app_cfg.llm.monthly_budget_usd} USD) đã sử dụng hết",
        )
        if app_cfg.llm.cache_enabled:
            write_cached_recommendation(app_cfg, canonical_dict, rec.model_dump())
        return rec

    # 4. Kiểm tra API key
    resolved_key = api_key or get_api_key_or_none(app_cfg.llm)
    if not resolved_key:
        log.info(
            "Không tìm thấy biến môi trường '%s'. Kích hoạt fallback template cho SKU %s.",
            app_cfg.llm.api_key_env,
            pack.sku,
        )
        rec = generate_fallback_recommendation(
            pack,
            reason=f"Chưa cấu hình API Key ({app_cfg.llm.api_key_env})",
        )
        if app_cfg.llm.cache_enabled:
            write_cached_recommendation(app_cfg, canonical_dict, rec.model_dump())
        return rec

    # 5. Gọi OpenAI API với retry guardrails
    messages = format_messages_for_llm(pack)
    max_retries = 2
    last_errors: list[str] = []

    client = http_client or httpx.Client(timeout=30.0)
    should_close_client = http_client is None

    try:
        for attempt in range(max_retries + 1):
            try:
                response = client.post(
                    OPENAI_CHAT_URL,
                    headers={
                        "Authorization": f"Bearer {resolved_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": app_cfg.llm.model,
                        "messages": messages,
                        "temperature": 0.1,
                        "max_tokens": 1500,
                        "response_format": {"type": "json_object"},
                    },
                )
                response.raise_for_status()
                res_data = response.json()

                # Ghi nhận usage
                usage = res_data.get("usage", {})
                p_tokens = usage.get("prompt_tokens", 0)
                c_tokens = usage.get("completion_tokens", 0)
                record_token_usage(app_cfg, p_tokens, c_tokens)

                # Phân tích cú pháp và kiểm định bằng chứng
                raw_text = res_data["choices"][0]["message"]["content"]
                rec, errors = parse_and_validate_recommendation(raw_text, pack)

                if errors:
                    log.warning(
                        "Lần thử %d cho SKU %s phát hiện lỗi validation: %s",
                        attempt + 1,
                        pack.sku,
                        errors,
                    )
                    last_errors = errors
                    # Bổ sung thông điệp nhắc nhở để LLM sửa sai ở lần sau
                    messages.append({"role": "assistant", "content": raw_text})
                    messages.append({
                        "role": "user",
                        "content": (
                            f"Phản hồi trước bị lỗi: {'; '.join(errors)}. "
                            "Hãy sửa lại: BẮT BUỘC chỉ trích dẫn review_id có trong danh sách và trả về JSON hợp lệ."
                        ),
                    })
                    continue

                assert rec is not None
                if app_cfg.llm.cache_enabled:
                    write_cached_recommendation(app_cfg, canonical_dict, rec.model_dump())
                return rec

            except Exception as e:
                log.warning("Lỗi gọi OpenAI API ở lần thử %d: %s", attempt + 1, e)
                last_errors = [str(e)]
    finally:
        if should_close_client:
            client.close()

    # 6. Fallback nếu sau các lần thử vẫn không thành công
    log.error(
        "Gọi LLM cho SKU %s thất bại sau %d lần thử. Kích hoạt fallback. Lỗi: %s",
        pack.sku,
        max_retries + 1,
        last_errors,
    )
    fallback_rec = generate_fallback_recommendation(
        pack,
        reason=f"LLM không khả dụng hoặc lỗi validation: {'; '.join(last_errors[:2])}",
    )
    if app_cfg.llm.cache_enabled:
        write_cached_recommendation(app_cfg, canonical_dict, fallback_rec.model_dump())
    return fallback_rec
