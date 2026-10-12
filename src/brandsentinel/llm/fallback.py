"""Mẫu khuyến nghị dự phòng theo quy tắc (Rule-based Fallback Template) cho Module LLM (M5).

Được kích hoạt khi:
- Không thể kết nối OpenAI API (lỗi mạng, timeout, mã lỗi 429/500).
- Vượt hạn mức ngân sách tháng 5.0 USD (circuit breaker).
- Phản hồi từ LLM vi phạm schema hoặc dính lỗi ảo giác trích dẫn sau khi đã thử lại (max_retries).

Đảm bảo hệ thống KHÔNG BAO GIỜ bị dừng đột ngột và luôn cung cấp khuyến nghị có cấu trúc chuẩn mực.
"""

from __future__ import annotations

from datetime import UTC, datetime

from brandsentinel.llm.schemas import (
    ActionItem,
    DepartmentActions,
    EvidencePack,
    PriorityLevel,
    ProbableCause,
    RecommendationOutput,
    SCCTCrisisType,
)


def generate_fallback_recommendation(
    pack: EvidencePack,
    *,
    reason: str = "LLM unavailable (Rule-based fallback activated)",
) -> RecommendationOutput:
    """Tạo khuyến nghị cứu nguy mang tính định lượng dựa trên luật nghiệp vụ SCCT."""
    triggered_ids = {ind.indicator_id for ind in pack.triggered_indicators}
    sample_review_ids = [r.review_id for r in pack.reviews[:3]]

    # Kịch bản 1: Nghi ngờ review bombing / thao túng đánh giá -> Cụm SCCT: Victim
    if pack.verify_flag or "I11" in triggered_ids or "I10" in triggered_ids:
        scct_type = SCCTCrisisType.VICTIM
        summary = (
            f"[DỰ PHÒNG - {reason}] Phát hiện bất thường về tỷ lệ đánh giá xác thực hoặc độ lệch rating-nội dung. "
            f"Nghi ngờ hiện tượng thao túng đánh giá hoặc tấn công review bombing đối với SKU {pack.sku}."
        )
        causes = [
            ProbableCause(
                cause="Bất thường về tỷ lệ verified purchase và độ lệch rating phản ánh nguy cơ review bombing.",
                evidence_review_ids=sample_review_ids,
                confidence=0.75,
            )
        ]
        actions = DepartmentActions(
            QA_KyThuat=[
                ActionItem(
                    priority=PriorityLevel.MEDIUM,
                    action="Kiểm tra nhật ký giao dịch và đối soát mã đơn hàng phát sinh các đánh giá nghi vấn.",
                    timeline="48 giờ",
                )
            ],
            VanHanh=[
                ActionItem(
                    priority=PriorityLevel.LOW,
                    action="Duy trì kiểm soát xuất nhập kho bình thường, ghi nhận các phản hồi thực tế từ đơn hoàn tất.",
                    timeline="7 ngày",
                )
            ],
            CSKH=[
                ActionItem(
                    priority=PriorityLevel.HIGH,
                    action="Gắn cờ các yêu cầu bồi thường từ tài khoản chưa xác thực để kiểm tra chéo trước khi duyệt chi.",
                    timeline="24 giờ",
                )
            ],
            TruyenThong=[
                ActionItem(
                    priority=PriorityLevel.MEDIUM,
                    action="Theo dõi diễn biến mạng xã hội; chưa phát ngôn rộng rãi để tránh tạo hiệu ứng truyền thông ảo.",
                    timeline="24-48 giờ",
                )
            ],
        )

    # Kịch bản 2: Báo động rủi ro an toàn, sức khỏe, pháp lý -> Cụm SCCT: Preventable
    elif "I8" in triggered_ids:
        scct_type = SCCTCrisisType.PREVENTABLE
        summary = (
            f"[DỰ PHÒNG - {reason}] CẢNH BÁO AN TOÀN NGHIÊM TRỌNG: Phát hiện các từ khóa liên quan đến nguy cơ chập cháy, "
            f"quá nhiệt, độc hại hoặc rủi ro sức khỏe người tiêu dùng tại SKU {pack.sku}."
        )
        causes = [
            ProbableCause(
                cause="Khuyết tật an toàn linh kiện hoặc khâu kiểm định chất lượng xuất xưởng bị lỗi lặp lại.",
                evidence_review_ids=sample_review_ids,
                confidence=0.90,
            )
        ]
        actions = DepartmentActions(
            QA_KyThuat=[
                ActionItem(
                    priority=PriorityLevel.HIGH,
                    action="Tạm dừng xuất kho ngay lập tức toàn bộ lô hàng sản xuất liên quan; thanh tra kiểm định an toàn kỹ thuật.",
                    timeline="Ngay lập tức (trong 12 giờ)",
                )
            ],
            VanHanh=[
                ActionItem(
                    priority=PriorityLevel.HIGH,
                    action="Cách ly các kiện hàng cùng batch ID tại tất cả các trung tâm phân phối và kho hoàn tất đơn.",
                    timeline="24 giờ",
                )
            ],
            CSKH=[
                ActionItem(
                    priority=PriorityLevel.HIGH,
                    action="Chủ động liên hệ các khách hàng đã gửi phản ánh sự cố an toàn; hỗ trợ hoàn tiền 100% và viện phí nếu có.",
                    timeline="24 giờ",
                )
            ],
            TruyenThong=[
                ActionItem(
                    priority=PriorityLevel.HIGH,
                    action="Kích hoạt kịch bản khủng hoảng Preventable; chuẩn bị thông cáo an toàn và hướng dẫn ngưng sử dụng khẩn cấp.",
                    timeline="24-48 giờ",
                )
            ],
        )

    # Kịch bản 3: Biến động chất lượng hoặc vận chuyển thông thường -> Cụm SCCT: Accidental
    else:
        scct_type = SCCTCrisisType.ACCIDENTAL
        summary = (
            f"[DỰ PHÒNG - {reason}] Ghi nhận sự sụt giảm trải nghiệm đột ngột và gia tăng phản hồi tiêu cực "
            f"của khách hàng tại SKU {pack.sku} (RiskScore: {pack.risk_score:.4f})."
        )
        causes = [
            ProbableCause(
                cause="Sự cố kỹ thuật đột xuất hoặc khâu vận chuyển/đóng gói phát sinh lỗi khuyết tật cục bộ.",
                evidence_review_ids=sample_review_ids,
                confidence=0.70,
            )
        ]
        actions = DepartmentActions(
            QA_KyThuat=[
                ActionItem(
                    priority=PriorityLevel.HIGH,
                    action="Lấy mẫu ngẫu nhiên 20 sản phẩm tại kho để kiểm tra độ hoàn thiện và chức năng cơ bản.",
                    timeline="48 giờ",
                )
            ],
            VanHanh=[
                ActionItem(
                    priority=PriorityLevel.MEDIUM,
                    action="Rà soát lại quy trình đóng gói chống sốc và làm việc với đơn vị vận chuyển.",
                    timeline="3 ngày",
                )
            ],
            CSKH=[
                ActionItem(
                    priority=PriorityLevel.HIGH,
                    action="Tăng cường nhân sự giải quyết các yêu cầu hoàn hàng đổi trả trong vòng 24-48 giờ.",
                    timeline="24-48 giờ",
                )
            ],
            TruyenThong=[
                ActionItem(
                    priority=PriorityLevel.LOW,
                    action="Cập nhật tài liệu câu hỏi thường gặp (FAQ) và hướng dẫn sử dụng sản phẩm trên các kênh chính thức.",
                    timeline="5 ngày",
                )
            ],
        )

    return RecommendationOutput(
        sku=pack.sku,
        category=pack.category,
        window_end=pack.window_end,
        risk_score=pack.risk_score,
        summary=summary,
        scct_crisis_type=scct_type,
        probable_causes=causes,
        actions=actions,
        generated_at=datetime.now(UTC).isoformat(),
        model_used="rule_based_template",
        is_fallback=True,
    )
