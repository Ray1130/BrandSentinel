"""Kiểm thử cho Stage 4: LLM Client, Budget Tracker, Caching và Pipeline E2E."""

import json
from datetime import date
from pathlib import Path
import httpx
import pytest

from brandsentinel.core.config import Config
from brandsentinel.core.io import read_json, write_table
from brandsentinel.core.types import Level, Table
from brandsentinel.llm.client import (
    call_llm_with_guardrails,
    is_budget_exceeded,
    read_budget_state,
    record_token_usage,
    require_api_key,
)
from brandsentinel.llm.schemas import (
    EvidencePack,
    IndicatorEvidence,
    RecommendationOutput,
    ReviewEvidence,
    SCCTCrisisType,
)
from brandsentinel.pipeline import run_explain, run_stage
from brandsentinel.testing.mock_data import make_all


@pytest.fixture
def sample_pack() -> EvidencePack:
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
            IndicatorEvidence(indicator_id="I1", name="Volume", triggered=True),
            IndicatorEvidence(indicator_id="I8", name="Safety", triggered=True),
        ],
        reviews=[
            ReviewEvidence(
                review_id="rev_1",
                rating=1,
                text="The charger overheated and melted!",
                date="2023-03-14",
                risk_hit=True,
                risk_terms=["overheat"],
            )
        ],
    )


def test_budget_tracker_calculation_and_circuit_breaker(cfg: Config, tmp_path: Path):
    """Kiểm tra tính chi phí token và kích hoạt circuit breaker khi vượt ngân sách."""
    isolated_cfg = cfg.model_copy(update={"root": tmp_path})

    # Ban đầu ngân sách = 0
    state_0 = read_budget_state(isolated_cfg)
    assert state_0["total_cost_usd"] == 0.0
    assert not is_budget_exceeded(isolated_cfg)

    # Ghi nhận 10,000 prompt tokens + 2,000 completion tokens
    # Cost = (10 * 0.00015) + (2 * 0.00060) = 0.0015 + 0.0012 = 0.0027 USD
    spent = record_token_usage(isolated_cfg, 10000, 2000)
    assert 0.0026 <= spent <= 0.0028

    state_1 = read_budget_state(isolated_cfg)
    assert state_1["total_prompt_tokens"] == 10000
    assert state_1["total_completion_tokens"] == 2000
    assert not is_budget_exceeded(isolated_cfg)

    # Giả lập tiêu 50 triệu token để vượt hạn mức 5.0 USD
    record_token_usage(isolated_cfg, 35000000, 5000000)
    assert is_budget_exceeded(isolated_cfg)


def test_call_llm_auto_fallback_when_budget_exceeded(sample_pack, cfg: Config, tmp_path: Path):
    """Khi ngân sách đã hết, tự động chuyển sang fallback template an toàn."""
    isolated_cfg = cfg.model_copy(update={"root": tmp_path})
    # Đẩy chi phí vượt 5.0 USD
    record_token_usage(isolated_cfg, 40000000, 5000000)

    rec = call_llm_with_guardrails(sample_pack, isolated_cfg)
    assert rec.is_fallback is True
    assert rec.sku == sample_pack.sku
    assert "ngân sách" in rec.summary.lower()


def test_call_llm_auto_fallback_when_no_api_key(sample_pack, cfg: Config, tmp_path: Path, monkeypatch):
    """Khi không có OPENAI_API_KEY trong môi trường, tự động fallback mà không crash."""
    isolated_cfg = cfg.model_copy(update={"root": tmp_path})
    monkeypatch.delenv(cfg.llm.api_key_env, raising=False)

    rec = call_llm_with_guardrails(sample_pack, isolated_cfg)
    assert rec.is_fallback is True
    assert rec.sku == sample_pack.sku
    assert "api key" in rec.summary.lower()


def test_call_llm_mock_http_success_and_cache(sample_pack, cfg: Config, tmp_path: Path):
    """Kiểm tra gọi mock HTTP response thành công, ghi token usage và lưu cache."""
    isolated_cfg = cfg.model_copy(update={"root": tmp_path})

    mock_llm_response = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "usage": {"prompt_tokens": 500, "completion_tokens": 150, "total_tokens": 650},
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": json.dumps({
                        "sku": sample_pack.sku,
                        "category": sample_pack.category,
                        "window_end": sample_pack.window_end,
                        "risk_score": sample_pack.risk_score,
                        "summary": "Phát hiện sự cố quá nhiệt pin sạc nghiêm trọng.",
                        "scct_crisis_type": "Preventable",
                        "probable_causes": [
                            {
                                "cause": "Lỗi thiết kế mạch tản nhiệt của củ sạc",
                                "evidence_review_ids": ["rev_1"],
                                "confidence": 0.95,
                            }
                        ],
                        "actions": {
                            "QA_KyThuat": [{"priority": "HIGH", "action": "Kiểm định tản nhiệt", "timeline": "12h"}],
                            "VanHanh": [{"priority": "HIGH", "action": "Dừng xuất kho lô tuần 34", "timeline": "24h"}],
                            "CSKH": [{"priority": "HIGH", "action": "Hoàn tiền khách gặp sự cố", "timeline": "24h"}],
                            "TruyenThong": [{"priority": "HIGH", "action": "Soạn thông cáo báo chí", "timeline": "24h"}],
                        },
                    }),
                }
            }
        ],
    }

    class MockTransport(httpx.BaseTransport):
        def handle_request(self, request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=mock_llm_response)

    client = httpx.Client(transport=MockTransport())

    # Gọi lần 1: qua mock HTTP
    rec1 = call_llm_with_guardrails(
        sample_pack,
        isolated_cfg,
        api_key="sk-test-mock-key",
        http_client=client,
    )
    assert rec1.is_fallback is False
    assert rec1.scct_crisis_type == SCCTCrisisType.PREVENTABLE
    assert rec1.probable_causes[0].evidence_review_ids == ["rev_1"]

    budget_after_call1 = read_budget_state(isolated_cfg)
    assert budget_after_call1["total_prompt_tokens"] == 500
    assert budget_after_call1["total_completion_tokens"] == 150

    # Gọi lần 2: phải trúng Cache hit (tokens không tăng thêm)
    rec2 = call_llm_with_guardrails(sample_pack, isolated_cfg, api_key="sk-test-mock-key")
    assert rec2.sku == rec1.sku
    assert rec2.summary == rec1.summary

    budget_after_call2 = read_budget_state(isolated_cfg)
    assert budget_after_call2["total_prompt_tokens"] == 500  # Không tăng vì cache hit


def test_pipeline_run_explain_end_to_end(cfg: Config, tmp_path: Path):
    """Kiểm tra tích hợp đầu cuối stage explain trong pipeline với mock data."""
    isolated_cfg = cfg.model_copy(update={"root": tmp_path})
    cat = "Mock_Category"

    # Tạo dữ liệu mock và lưu vào thư mục isolated
    tables = make_all(seed=42, n_skus=6, days=120, category=cat, cfg=isolated_cfg)
    write_table(tables[Table.CLEAN_REVIEWS], Table.CLEAN_REVIEWS, isolated_cfg)
    write_table(tables[Table.ALERTS], Table.ALERTS, isolated_cfg)
    write_table(tables[Table.NLP_FEATURES], Table.NLP_FEATURES, isolated_cfg)

    # Chạy stage explain qua run_explain
    recs = run_explain(cat, config=isolated_cfg, force_fallback=True)
    assert len(recs) > 0

    # Kiểm tra cấu trúc output khuyến nghị
    first_rec = recs[0]
    assert "sku" in first_rec
    assert "actions" in first_rec
    assert "QA_KyThuat" in first_rec["actions"]
    assert "VanHanh" in first_rec["actions"]
    assert "CSKH" in first_rec["actions"]
    assert "TruyenThong" in first_rec["actions"]

    # Kiểm tra file tổng hợp recommendations.json đã được ghi ra đĩa
    rec_file = (
        isolated_cfg.path("data_processed")
        / "recommendations"
        / f"category={cat}"
        / "recommendations.json"
    )
    assert rec_file.is_file()
    saved_data = read_json(rec_file)
    assert len(saved_data) == len(recs)

    # Chạy lại qua run_stage("explain", ...)
    summary = run_stage("explain", cat, config=isolated_cfg)
    assert summary["recommendations"] == len(recs)
    assert summary["skus"] > 0
