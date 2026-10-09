"""NLP pipeline orchestration module for BrandSentinel.

Ghép toàn bộ các thành phần NLP hiện có:
- Sentiment analysis & mismatch detection (Task 2)
- Risk Layer 1: Rule-based lexicon (Task 3)
- Risk Layer 2: Semantic embedding similarity (Phase 2)
- Aspect extraction & aspect sentiment (Task 4)

Tạo bảng kết quả hoàn chỉnh tuân thủ hợp đồng dữ liệu Table.NLP_FEATURES.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import polars as pl

from brandsentinel.core.config import Config, get_config
from brandsentinel.core.schemas import empty_table, validate
from brandsentinel.core.types import Table
from brandsentinel.features.nlp.aspect import (
    AspectExtractor,
    compute_aspect,
    get_aspect_extractor,
)
from brandsentinel.features.nlp.embedder import (
    RiskEmbedder,
    compute_i8_combined,
    get_risk_embedder,
)
from brandsentinel.features.nlp.risk_lexicon import (
    RiskLexiconDetector,
    get_risk_detector,
)
from brandsentinel.features.nlp.sentiment import (
    SentimentAnalyzer,
    compute_sentiment,
    get_sentiment_analyzer,
)

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)

ORDERED_NLP_COLUMNS = [
    "review_id",
    "category",
    "date",
    "sentiment_score",
    "risk_hit",
    "risk_terms",
    "risk_sim",
    "aspect",
    "aspect_sentiment",
    "mismatch",
]


def compute_nlp_features(
    clean_reviews: pl.DataFrame,
    cfg: Config | None = None,
    *,
    analyzer: SentimentAnalyzer | None = None,
    detector: RiskLexiconDetector | None = None,
    embedder: RiskEmbedder | None = None,
    extractor: AspectExtractor | None = None,
    risk_similarity_threshold: float | None = None,
    validate_output: bool = True,
) -> pl.DataFrame:
    """Orchestrate full NLP feature extraction from clean_reviews into Table.NLP_FEATURES.

    Args:
        clean_reviews: DataFrame chứa ít nhất [review_id, category, date, text_norm / text_raw, rating].
        cfg: Cấu hình hệ thống (tùy chọn, mặc định lấy get_config()).
        analyzer: SentimentAnalyzer tái sử dụng (tùy chọn).
        detector: RiskLexiconDetector tái sử dụng (tùy chọn).
        embedder: RiskEmbedder tái sử dụng (tùy chọn).
        extractor: AspectExtractor tái sử dụng (tùy chọn).
        risk_similarity_threshold: Ngưỡng tương đồng rủi ro cho Layer 2 (mặc định đọc từ config).
        validate_output: Có chạy schema validator Pandera trước khi trả kết quả hay không.

    Returns:
        DataFrame tuân thủ 100% schema và semantic rules của Table.NLP_FEATURES.
    """
    if clean_reviews.is_empty():
        empty_df = empty_table(Table.NLP_FEATURES)
        return validate(Table.NLP_FEATURES, empty_df) if validate_output else empty_df

    # Kiểm tra các cột bắt buộc
    required_cols = {"review_id", "category", "date"}
    missing = [c for c in required_cols if c not in clean_reviews.columns]
    if missing:
        raise ValueError(f"clean_reviews thiếu cột bắt buộc cho nlp_features: {missing}")

    if "text_norm" not in clean_reviews.columns and "text_raw" not in clean_reviews.columns:
        raise ValueError("clean_reviews phải có ít nhất một trong hai cột 'text_norm' hoặc 'text_raw'")

    # Kiểm tra tính duy nhất của review_id
    n_total = clean_reviews.height
    n_unique = clean_reviews["review_id"].n_unique()
    if n_unique != n_total:
        raise ValueError(
            f"clean_reviews có review_id bị trùng lặp: {n_total} dòng nhưng chỉ có {n_unique} unique review_id"
        )

    active_cfg = cfg or get_config()

    # 1. Khởi tạo/tái sử dụng các thành phần xử lý
    # Tái sử dụng cùng một analyzer cho cả Sentiment và Aspect để không tải lại mô hình nặng vào RAM
    active_analyzer = analyzer or get_sentiment_analyzer(cfg=active_cfg)
    active_detector = detector or get_risk_detector(cfg=active_cfg)
    active_embedder = embedder or get_risk_embedder(cfg=active_cfg)
    active_extractor = extractor or get_aspect_extractor(
        cfg=active_cfg, sentiment_analyzer=active_analyzer
    )

    log.info("Chạy full NLP pipeline trên %d reviews...", n_total)

    # 2. Bước 1: Phân tích cảm xúc & mismatch (Task 2)
    sent_df = compute_sentiment(clean_reviews, cfg=active_cfg, analyzer=active_analyzer)

    # 3. Bước 2: Phát hiện rủi ro kết hợp Layer 1 Lexicon + Layer 2 Embedding (Task 3 + Phase 2)
    risk_df = compute_i8_combined(
        clean_reviews,
        cfg=active_cfg,
        detector=active_detector,
        embedder=active_embedder,
        threshold=risk_similarity_threshold,
        mode="combined",
    )

    # 4. Bước 3: Phân loại khía cạnh & aspect sentiment (Task 4)
    aspect_df = compute_aspect(clean_reviews, cfg=active_cfg, extractor=active_extractor)

    # 5. Bước 4: Ghép nối tất cả các bảng bằng review_id (Join contract)
    base_meta = clean_reviews.select(["review_id", "category", "date"])

    nlp_features = (
        base_meta.join(
            sent_df.select(["review_id", "sentiment_score", "mismatch"]),
            on="review_id",
            how="left",
        )
        .join(
            risk_df.select(["review_id", "risk_hit", "risk_terms", "risk_sim"]),
            on="review_id",
            how="left",
        )
        .join(
            aspect_df.select(["review_id", "aspect", "aspect_sentiment"]),
            on="review_id",
            how="left",
        )
    )

    # Đảm bảo đúng thứ tự và danh sách cột trong hợp đồng
    result = nlp_features.select(ORDERED_NLP_COLUMNS)

    # Kiểm tra bảo toàn dữ liệu
    if result.height != n_total:
        raise RuntimeError(
            f"Lỗi join contract: số dòng đầu vào ({n_total}) khác số dòng kết quả ({result.height})"
        )

    # 6. Bước 5: Validate chính thức bằng schema của project
    if validate_output:
        result = validate(Table.NLP_FEATURES, result)

    log.info("Hoàn thành full NLP pipeline: %d dòng hợp lệ theo Table.NLP_FEATURES", result.height)
    return result
