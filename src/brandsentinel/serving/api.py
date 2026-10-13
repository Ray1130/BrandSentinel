"""FastAPI serving module for BrandSentinel.

Cung cấp các endpoint phục vụ API quản trị rủi ro & khuyến nghị LLM:
- POST /api/v1/analyze-risk: Nhận EvidencePack, gọi LLM qua Guardrails/Cache, trả về RecommendationOutput.
- GET  /api/v1/alerts: Truy vấn danh sách cảnh báo (lọc theo category, level, limit).
- GET  /api/v1/recommendations: Truy vấn danh sách khuyến nghị SCCT đã được sinh.
- GET  /api/v1/health: Kiểm tra trạng thái hoạt động của service.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
import polars as pl

from brandsentinel.core.config import Config, get_config, load_config
from brandsentinel.core.io import read_json, read_table
from brandsentinel.core.types import Table
from brandsentinel.llm.client import call_llm_with_guardrails
from brandsentinel.llm.schemas import EvidencePack, RecommendationOutput

log = logging.getLogger(__name__)


def create_app(config: Config | None = None) -> FastAPI:
    """Tạo và cấu hình ứng dụng FastAPI cho BrandSentinel."""
    cfg = config or load_config()

    app = FastAPI(
        title="BrandSentinel API",
        version="0.1.0",
        description="Early Warning of Brand Crises via Temporal Anomaly and Change-Point Detection on Online Reviews",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Cấu hình CORS để Dashboard Streamlit / React có thể gọi trực tiếp
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/v1/health", tags=["System"])
    @app.get("/health", tags=["System"])
    def health_check() -> dict[str, Any]:
        """Endpoint kiểm tra tình trạng dịch vụ (Health Check)."""
        return {
            "status": "ok",
            "service": "BrandSentinel",
            "version": "0.1.0",
            "timestamp": datetime.now(UTC).isoformat(),
        }

    @app.post(
        "/api/v1/analyze-risk",
        response_model=RecommendationOutput,
        status_code=status.HTTP_200_OK,
        tags=["LLM Analysis"],
        summary="Phân tích rủi ro và sinh khuyến nghị SCCT từ gói bằng chứng EvidencePack",
    )
    def analyze_risk(
        evidence_pack: EvidencePack,
        force_fallback: bool = Query(
            False,
            description="Bật cờ này để ép buộc dùng Rule-based Fallback template (dùng khi offline / test)",
        ),
    ) -> RecommendationOutput:
        """Nhận gói bằng chứng của SKU đạt mức HIGH, kích hoạt LLM với đầy đủ Guardrails:

        - 1. Kiểm tra Cache SHA-256 (0 chi phí nếu đã phân tích cùng bằng chứng).
        - 2. Kiểm tra Circuit Breaker ngân sách (chặn nếu vượt 5 USD).
        - 3. Gọi OpenAI API với Timeout (30s) và Retry tối đa 2 lần nếu trích dẫn sai.
        - 4. Tự động chuyển Fallback an toàn nếu LLM lỗi.
        """
        try:
            rec = call_llm_with_guardrails(
                pack=evidence_pack,
                cfg=cfg,
                force_fallback=force_fallback,
            )
            return rec
        except Exception as e:
            log.exception("Lỗi khi xử lý analyze-risk cho SKU %s: %s", evidence_pack.sku, e)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Lỗi phân tích rủi ro cho SKU {evidence_pack.sku}: {e}",
            ) from e

    @app.get(
        "/api/v1/alerts",
        tags=["Alerts"],
        summary="Truy vấn danh sách cảnh báo từ bảng alerts",
    )
    def get_alerts(
        category: str = Query(..., description="Ngành hàng, ví dụ: Home_and_Kitchen"),
        level: str | None = Query(
            None,
            description="Lọc theo mức rủi ro: HIGH, MEDIUM, LOW",
        ),
        limit: int = Query(50, ge=1, le=500, description="Số lượng bản ghi tối đa"),
    ) -> list[dict[str, Any]]:
        """Đọc và trả về danh sách cảnh báo từ Table.ALERTS."""
        try:
            alerts_df = read_table(Table.ALERTS, cfg, category=category)
        except FileNotFoundError:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Chưa có dữ liệu alerts cho category '{category}'. Hãy chạy stage score trước.",
            ) from None
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Lỗi khi đọc bảng alerts: {e}",
            ) from e

        if alerts_df.is_empty():
            return []

        if level:
            alerts_df = alerts_df.filter(pl.col("level") == level.upper())

        alerts_df = alerts_df.sort(["score", "window_end"], descending=[True, True]).head(limit)
        return alerts_df.to_dicts()

    @app.get(
        "/api/v1/recommendations",
        tags=["Recommendations"],
        summary="Truy vấn danh sách khuyến nghị SCCT đã được sinh từ stage explain",
    )
    def get_recommendations(
        category: str = Query(..., description="Ngành hàng, ví dụ: Home_and_Kitchen"),
        sku: str | None = Query(None, description="Lọc theo mã SKU cụ thể"),
    ) -> list[dict[str, Any]]:
        """Đọc danh sách khuyến nghị từ recommendations.json."""
        summary_path = (
            cfg.path("data_processed")
            / "recommendations"
            / f"category={category}"
            / "recommendations.json"
        )
        if not summary_path.is_file():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Chưa có recommendations.json cho category '{category}'. Hãy chạy stage explain trước.",
            )

        try:
            recs = read_json(summary_path)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Không thể đọc file recommendations.json: {e}",
            ) from e

        if not isinstance(recs, list):
            recs = [recs]

        if sku:
            recs = [r for r in recs if r.get("sku") == sku]

        return recs

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("brandsentinel.serving.api:app", host="0.0.0.0", port=8000, reload=True)
