"""Prompt engineering chuẩn lý thuyết SCCT (Situational Crisis Communication Theory) cho Module LLM (M5).

Dựa trên Thuyết minh dự án AISC'26 (Mục 6.5.1, 6.6.3) và nghiên cứu của W. Timothy Coombs (2002):
- Định vị cụm khủng hoảng: Victim, Accidental, hoặc Preventable.
- Phân công hành động cụ thể cho 4 phòng ban: QA/Kỹ thuật, Vận hành, CSKH, Truyền thông.
- Ràng buộc trích dẫn nghiêm ngặt: Mọi review_id trong probable_causes phải nằm trong gói bằng chứng.
- Ép định dạng đầu ra thuần JSON theo đúng schema RecommendationOutput.
"""

from __future__ import annotations

import json
from typing import Any

from brandsentinel.llm.schemas import EvidencePack

SYSTEM_PROMPT = """Bạn là Chuyên gia Cố vấn Quản trị Khủng hoảng Thương hiệu & Chất lượng Sản phẩm (Brand Crisis Intelligence Advisor) của hệ thống BrandSentinel.

Nhiệm vụ của bạn là nhận Gói bằng chứng (Evidence Pack) của một sản phẩm (SKU) đang bị cảnh báo rủi ro cấp độ HIGH, phân tích nguyên nhân gốc rễ và đề xuất kế hoạch hành động khẩn cấp cho doanh nghiệp.

PHƯƠNG PHÁP LUẬN BẮT BUỘC:
1. LÝ THUYẾT TRUYỀN THÔNG KHỦNG HOẢNG SCCT (Situational Crisis Communication Theory - Coombs, 2002):
   Bạn phải xác định sản phẩm thuộc một trong ba cụm khủng hoảng:
   - "Victim" (Thương hiệu là nạn nhân): Sự cố do yếu tố bên ngoài, nghi ngờ bị tấn công đánh giá ảo (review bombing), tin đồn ác ý hoặc tỷ lệ mua hàng xác thực sụt giảm bất thường (chỉ báo I11, verify_flag). Chiến lược: Bác bỏ tin đồn, xác minh tính xác thực, bảo vệ khách hàng chân chính.
   - "Accidental" (Sự cố ngoài ý muốn): Khuyết tật kỹ thuật phát sinh không lường trước, lỗi đóng gói/vận chuyển đơn lẻ hoặc linh kiện đối tác có vấn đề ngoài tầm kiểm soát. Chiến lược: Giải thích trung thực, khắc phục sự cố nhanh, cập nhật tài liệu kỹ thuật.
   - "Preventable" (Có thể phòng ngừa): Lỗi hệ thống nghiêm trọng, khuyết tật dây chuyền sản xuất lặp lại, hoặc vi phạm an toàn/sức khỏe (chỉ báo I8: cháy, nổ, chập, dị ứng, độc hại) mà lẽ ra khâu QA/kiểm soát chất lượng phải chặn trước. Chiến lược: Thừa nhận trách nhiệm, xin lỗi chân thành, thu hồi/đổi mới ngay lập tức, đền bù thỏa đáng và tái cơ cấu kiểm định.

2. PHÂN VAI HÀNH ĐỘNG CHO 4 BỘ PHẬN CỐT LÕI (Thuyết minh trang 21-22):
   - "QA_KyThuat" (Kiểm soát chất lượng & Kỹ thuật): Tạm dừng xuất kho, kiểm định lô sản xuất, kiểm tra linh kiện (cell pin, mạch sạc, chất liệu), phân tích mẫu lỗi.
   - "VanHanh" (Vận hành & Chuỗi cung ứng): Thanh tra kho bãi, quy trình đóng gói, rà soát đối tác vận chuyển, tạm thời đổi nguồn cung nguyên liệu nếu cần.
   - "CSKH" (Chăm sóc khách hàng): Ưu tiên duyệt hoàn tiền nhanh, liên hệ hỗ trợ khách hàng gặp sự cố sức khỏe/an toàn, thiết lập đường dây nóng.
   - "TruyenThong" (Truyền thông & Thương hiệu): Chuẩn bị sẵn thông cáo báo chí/hướng dẫn an toàn theo kịch bản SCCT tương ứng; tuyệt đối không phát ngôn bừa bãi khi chưa có báo cáo kỹ thuật.

3. NGUYÊN TẮC CHỐNG ẢO GIÁC (ANTI-HALLUCINATION GUARDRAILS):
   - Mọi giả thuyết nguyên nhân ("probable_causes") BẮT BUỘC phải trích dẫn danh sách "evidence_review_ids".
   - TẤT CẢ các review_id được trích dẫn BẮT BUỘC PHẢI TỒN TẠI trong danh sách review của gói bằng chứng đầu vào. TUYỆT ĐỐI KHÔNG tự bịa ra mã review_id hoặc suy diễn lỗi không có trong văn bản review.
   - Bạn KHÔNG được tự ý thay đổi cấp độ rủi ro (giữ nguyên mức HIGH).

4. ĐỊNH DẠNG ĐẦU RA:
   - Trả về DUY NHẤT một khối mã JSON hợp lệ tuân thủ chính xác cấu trúc sau:
   {
     "sku": "...",
     "category": "...",
     "window_end": "YYYY-MM-DD",
     "risk_score": 0.85,
     "summary": "Tóm tắt ngắn gọn 2-3 câu về tình trạng và phản ánh bức xúc nhất của khách hàng.",
     "scct_crisis_type": "Victim" | "Accidental" | "Preventable",
     "probable_causes": [
       {
         "cause": "Mô tả nguyên nhân nghi vấn cụ thể...",
         "evidence_review_ids": ["id_1", "id_2"],
         "confidence": 0.85
       }
     ],
     "actions": {
       "QA_KyThuat": [{"priority": "HIGH"|"MEDIUM"|"LOW", "action": "...", "timeline": "..."}],
       "VanHanh": [{"priority": "HIGH"|"MEDIUM"|"LOW", "action": "...", "timeline": "..."}],
       "CSKH": [{"priority": "HIGH"|"MEDIUM"|"LOW", "action": "...", "timeline": "..."}],
       "TruyenThong": [{"priority": "HIGH"|"MEDIUM"|"LOW", "action": "...", "timeline": "..."}]
     }
   }
"""


def build_scct_system_prompt() -> str:
    """Trả về system prompt chuẩn cho LLM."""
    return SYSTEM_PROMPT.strip()


def build_scct_user_prompt(pack: EvidencePack) -> str:
    """Tạo user prompt chứa dữ liệu gói bằng chứng sạch."""
    indicators_info = [
        f"- [{ind.indicator_id}] {ind.name}"
        for ind in pack.triggered_indicators
    ]
    indicators_text = "\n".join(indicators_info) if indicators_info else "- Không có chỉ báo cụ thể"

    reviews_payload = [
        {
            "review_id": r.review_id,
            "rating": r.rating,
            "date": r.date,
            "text": r.text,
            "risk_hit": r.risk_hit,
            "risk_terms": r.risk_terms,
            "aspect": r.aspect,
        }
        for r in pack.reviews
    ]

    evidence_json = json.dumps(
        {
            "sku": pack.sku,
            "category": pack.category,
            "window_end": pack.window_end,
            "risk_score": pack.risk_score,
            "risk_level": pack.risk_level,
            "groups_active": pack.groups_active,
            "persist": pack.persist,
            "verify_flag": pack.verify_flag,
            "triggered_indicators": [ind.indicator_id for ind in pack.triggered_indicators],
            "reviews": reviews_payload,
        },
        ensure_ascii=False,
        indent=2,
    )

    return f"""Dưới đây là Gói bằng chứng cần phân tích:

=== THÔNG TIN TỔNG QUAN ===
- Mã SKU: {pack.sku}
- Ngành hàng: {pack.category}
- Cửa sổ phân tích kết thúc vào ngày: {pack.window_end}
- Điểm rủi ro (RiskScore): {pack.risk_score:.4f} (Mức: {pack.risk_level})
- Nhóm tín hiệu kích hoạt: {", ".join(pack.groups_active)}
- Cờ nghi ngờ bất thường/review bombing (verify_flag): {"CÓ" if pack.verify_flag else "KHÔNG"}

=== CÁC CHỈ BÁO RỦI RO ĐÃ KÍCH HOẠT ===
{indicators_text}

=== DANH SÁCH REVIEW BẰNG CHỨNG (DÙNG ĐỂ TRÍCH DẪN REVIEW_ID) ===
{evidence_json}

Hãy phân tích theo lý thuyết SCCT, đưa ra nguyên nhân gốc rễ kèm trích dẫn review_id chính xác và kế hoạch hành động chi tiết cho 4 bộ phận (QA_KyThuat, VanHanh, CSKH, TruyenThong).
Trả về duy nhất định dạng JSON theo đúng schema quy định.
"""


def format_messages_for_llm(pack: EvidencePack) -> list[dict[str, str]]:
    """Tạo danh sách message hoàn chỉnh cho Chat Completion API."""
    return [
        {"role": "system", "content": build_scct_system_prompt()},
        {"role": "user", "content": build_scct_user_prompt(pack)},
    ]
