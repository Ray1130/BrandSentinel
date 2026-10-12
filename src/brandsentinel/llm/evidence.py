"""Bộ trích lọc và đóng gói bằng chứng (Evidence Pack Generator) cho Module LLM (M5).

Tuân thủ Thuyết minh dự án AISC'26 (Mục 6.6.3) và docs/data_contract.md:
1. Lọc cảnh báo theo cổng llm_gate (chỉ xử lý các SKU đạt mức HIGH).
2. Lấy dữ liệu đánh giá tương ứng trong cửa sổ W ngày [window_end - W + 1, window_end].
3. Thuật toán chọn lọc top-k review đại diện:
   - Ưu tiên review có từ khóa an toàn / rủi ro (risk_hit == True, I8).
   - Ưu tiên review tiêu cực (rating <= 2 sao).
   - Khử trùng lặp nội dung văn bản để đảm bảo tính đa dạng.
   - Ưu tiên review mới nhất trong cửa sổ.
4. Che thông tin cá nhân (PII Masking: email, số điện thoại, URL).
5. Đóng gói đối tượng EvidencePack chuẩn hóa cho cache hashing và LLM prompting.
"""

from __future__ import annotations

import re
from datetime import date, timedelta
from typing import Any

import polars as pl

from brandsentinel.core.config import Config, get_config
from brandsentinel.llm.schemas import (
    EvidencePack,
    IndicatorEvidence,
    ReviewEvidence,
)

# Mô tả chỉ báo rủi ro tiếng Việt chuẩn theo Thuyết minh dự án
INDICATOR_DESCRIPTIONS: dict[str, str] = {
    "I1": "Đột biến lượng review sau khử mùa vụ STL",
    "I2": "Tốc độ tăng review bất thường trong nhiều mốc liên tiếp",
    "I3": "Xu hướng gia tăng phản hồi tiêu cực kéo dài",
    "I4": "Gia tăng tỷ lệ đánh giá thấp (1-2 sao)",
    "I5": "Biến động rating kéo dài bất thường",
    "I6": "Phân cực phân phối rating (dồn mạnh về 1 sao và 5 sao)",
    "I7": "Dịch chuyển cụm chủ đề phàn nàn sang vấn đề mới",
    "I8": "Xuất hiện nội dung liên quan an toàn, sức khỏe hoặc pháp lý",
    "I9": "Mức độ tiêu cực tăng vọt theo khía cạnh cụ thể",
    "I10": "Rating và nội dung văn bản lệch pha (khen nhưng chấm 1 sao hoặc ngược lại)",
    "I11": "Tỷ lệ verified purchase sụt giảm bất thường khi volume tăng (nghi ngờ review bombing)",
}

# Regex che thông tin nhạy cảm (PII)
_RE_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
_RE_PHONE = re.compile(r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}")
_RE_URL = re.compile(r"https?://\S+|www\.\S+")


def mask_pii(text: str | None) -> str:
    """Che email, số điện thoại và URL trong nội dung review để bảo vệ quyền riêng tư."""
    if not text:
        return ""
    text = _RE_EMAIL.sub("[EMAIL]", text)
    text = _RE_PHONE.sub("[PHONE]", text)
    text = _RE_URL.sub("[URL]", text)
    return text.strip()


def _calculate_review_priority_score(row: dict[str, Any]) -> float:
    """Chấm điểm ưu tiên của một review để chọn top-k đại diện cho LLM.

    Điểm càng cao thì càng mang nhiều thông tin rủi ro đắt giá:
    - risk_hit (an toàn/sức khỏe): +10.0
    - Rating 1 sao: +5.0, 2 sao: +4.0, 3 sao: +1.0
    - Độ dài văn bản > 30 ký tự (nhiều chi tiết): +1.0
    - Mua hàng xác thực (verified): +0.5
    """
    score = 0.0
    if row.get("risk_hit") is True:
        score += 10.0

    rating = row.get("rating", 3)
    if rating == 1:
        score += 5.0
    elif rating == 2:
        score += 4.0
    elif rating == 3:
        score += 1.0

    text = row.get("text_norm") or row.get("text_raw") or ""
    if len(text) >= 30:
        score += 1.0

    if row.get("verified_purchase") is True:
        score += 0.5

    return score


def select_representative_reviews(
    reviews_df: pl.DataFrame,
    *,
    max_reviews: int = 10,
    mask_pii_enabled: bool = True,
) -> list[ReviewEvidence]:
    """Chọn lọc top-k review đại diện nhất từ tập review của một cửa sổ SKU.

    Loại bỏ review trùng nội dung và ưu tiên các phản hồi mang tính cảnh báo cao.
    """
    if reviews_df.is_empty():
        return []

    # Loại bỏ spam trước (nếu có cột is_spam)
    if "is_spam" in reviews_df.columns:
        valid_df = reviews_df.filter(~pl.col("is_spam").fill_null(False))
        if valid_df.is_empty():
            valid_df = reviews_df  # Fallback nếu toàn bộ bị gắn cờ
    else:
        valid_df = reviews_df

    rows = valid_df.to_dicts()

    # Tính điểm ưu tiên và sắp xếp giảm dần theo điểm và ngày mới nhất
    def sort_key(r: dict[str, Any]) -> tuple[float, str]:
        p_score = _calculate_review_priority_score(r)
        d_val = str(r.get("date", ""))
        return (p_score, d_val)

    sorted_rows = sorted(rows, key=sort_key, reverse=True)

    # Chọn lọc top-k với cơ chế khử trùng lặp nội dung
    selected: list[ReviewEvidence] = []
    seen_texts: set[str] = set()

    for r in sorted_rows:
        raw_text = r.get("text_norm") or r.get("text_raw") or ""
        cleaned_text = mask_pii(raw_text) if mask_pii_enabled else raw_text.strip()
        if not cleaned_text:
            continue

        # Chuẩn hóa ngắn để kiểm tra trùng lặp
        short_key = " ".join(cleaned_text.lower().split()[:15])
        if short_key in seen_texts:
            continue
        seen_texts.add(short_key)

        risk_terms = r.get("risk_terms")
        if risk_terms is None:
            terms_list = []
        elif isinstance(risk_terms, (list, tuple)):
            terms_list = [str(t) for t in risk_terms]
        else:
            terms_list = [str(risk_terms)]

        review_item = ReviewEvidence(
            review_id=str(r["review_id"]),
            rating=int(r.get("rating", 1)),
            text=cleaned_text,
            date=str(r.get("date", "")),
            verified_purchase=r.get("verified_purchase"),
            risk_hit=bool(r.get("risk_hit", False)),
            risk_terms=terms_list,
            aspect=str(r["aspect"]) if r.get("aspect") is not None else None,
        )
        selected.append(review_item)

        if len(selected) >= max_reviews:
            break

    # Nếu sau khi lọc trùng mà vẫn thiếu, lấy thêm các review khả dụng khác
    if not selected and rows:
        r = rows[0]
        selected.append(
            ReviewEvidence(
                review_id=str(r["review_id"]),
                rating=int(r.get("rating", 1)),
                text=mask_pii(r.get("text_norm") or r.get("text_raw") or "No content"),
                date=str(r.get("date", "")),
                verified_purchase=r.get("verified_purchase"),
                risk_hit=bool(r.get("risk_hit", False)),
            )
        )

    return selected


def build_evidence_pack(
    alert_record: dict[str, Any],
    clean_reviews_df: pl.DataFrame,
    *,
    nlp_features_df: pl.DataFrame | None = None,
    window_days: int = 7,
    max_reviews: int = 10,
    mask_pii_enabled: bool = True,
) -> EvidencePack:
    """Đóng gói EvidencePack cho một SKU tại mốc alert_record."""
    sku = str(alert_record["sku"])
    category = str(alert_record.get("category", "General"))
    window_end_val = alert_record["window_end"]

    if isinstance(window_end_val, (date,)):
        window_end_date = window_end_val
        window_end_str = window_end_val.isoformat()
    else:
        window_end_str = str(window_end_val)
        window_end_date = date.fromisoformat(window_end_str)

    window_start_date = window_end_date - timedelta(days=window_days - 1)

    # Lọc review trong cửa sổ thời gian
    sku_reviews = clean_reviews_df.filter(
        (pl.col("sku") == sku)
        & (pl.col("date") >= window_start_date)
        & (pl.col("date") <= window_end_date)
    )

    # Nếu có bảng nlp_features, kết nối thông tin cảm xúc & rủi ro
    if nlp_features_df is not None and not nlp_features_df.is_empty() and not sku_reviews.is_empty():
        cols_to_join = [
            c
            for c in ["review_id", "risk_hit", "risk_terms", "aspect", "sentiment_score"]
            if c in nlp_features_df.columns
        ]
        if len(cols_to_join) > 1:
            sku_reviews = sku_reviews.join(
                nlp_features_df.select(cols_to_join),
                on="review_id",
                how="left",
            )

    # Chọn top-k review đại diện
    selected_reviews = select_representative_reviews(
        sku_reviews,
        max_reviews=max_reviews,
        mask_pii_enabled=mask_pii_enabled,
    )

    # Chuẩn hóa danh sách chỉ báo kích hoạt
    triggered_ids = alert_record.get("triggered_ids") or []
    if isinstance(triggered_ids, (list, tuple)):
        indicator_list = [
            IndicatorEvidence(
                indicator_id=str(iid),
                name=INDICATOR_DESCRIPTIONS.get(str(iid), f"Chỉ báo {iid}"),
                triggered=True,
            )
            for iid in triggered_ids
        ]
    else:
        indicator_list = []

    # Nhóm tín hiệu active
    groups_active = alert_record.get("groups_active") or []
    if not isinstance(groups_active, list):
        groups_active = list(groups_active)

    return EvidencePack(
        sku=sku,
        category=category,
        window_end=window_end_str,
        risk_score=float(alert_record.get("score", 0.0)),
        risk_level="HIGH",
        groups_active=groups_active,
        persist=int(alert_record.get("persist", 0)),
        verify_flag=bool(alert_record.get("verify_flag", False)),
        triggered_indicators=indicator_list,
        reviews=selected_reviews,
        summary_stats={
            "window_days": window_days,
            "window_start": window_start_date.isoformat(),
            "total_reviews_in_window": sku_reviews.height,
        },
    )


def extract_high_alerts_evidence_packs(
    alerts_df: pl.DataFrame,
    clean_reviews_df: pl.DataFrame,
    *,
    nlp_features_df: pl.DataFrame | None = None,
    cfg: Config | None = None,
    min_level: str = "HIGH",
) -> list[EvidencePack]:
    """Cổng chính: Lọc các alert đạt mức min_level (mặc định HIGH) và đóng gói EvidencePack."""
    if alerts_df.is_empty():
        return []

    # Cổng lọc cấp độ rủi ro (llm_gate)
    high_alerts = alerts_df.filter(pl.col("level") == min_level)
    if high_alerts.is_empty():
        return []

    app_cfg = cfg or get_config()
    window_days = app_cfg.default.time.window_days

    # Đọc cấu hình max review và PII
    max_reviews = 10
    mask_pii_val = True
    if hasattr(app_cfg, "llm") and hasattr(app_cfg.llm, "max_top_reviews"):
        max_reviews = getattr(app_cfg.llm, "max_top_reviews", 10)
        mask_pii_val = getattr(app_cfg.llm, "mask_pii", True)

    evidence_packs: list[EvidencePack] = []
    for record in high_alerts.to_dicts():
        pack = build_evidence_pack(
            alert_record=record,
            clean_reviews_df=clean_reviews_df,
            nlp_features_df=nlp_features_df,
            window_days=window_days,
            max_reviews=max_reviews,
            mask_pii_enabled=mask_pii_val,
        )
        evidence_packs.append(pack)

    return evidence_packs


def canonical_evidence_dict(pack: EvidencePack) -> dict[str, Any]:
    """Chuyển EvidencePack sang dictionary chuẩn để hash SHA-256 (tương thích cache.py)."""
    return {
        "sku": pack.sku,
        "category": pack.category,
        "window_end": pack.window_end,
        "risk_score": round(pack.risk_score, 4),
        "risk_level": pack.risk_level,
        "groups_active": sorted(pack.groups_active),
        "persist": pack.persist,
        "verify_flag": pack.verify_flag,
        "triggered_indicators": [ind.indicator_id for ind in pack.triggered_indicators],
        "reviews": [
            {
                "id": r.review_id,
                "rating": r.rating,
                "text": r.text,
                "risk_hit": r.risk_hit,
            }
            for r in pack.reviews
        ],
    }
