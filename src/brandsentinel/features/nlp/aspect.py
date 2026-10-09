"""Aspect extraction and aspect-level sentiment analysis module for BrandSentinel (I9).

Phân loại review vào 4 khía cạnh chuẩn (quality, delivery, safety, refund)
và tính điểm cảm xúc riêng cho khía cạnh chủ đạo (dominant aspect).
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import yaml

from brandsentinel.core.config import Config, get_config
from brandsentinel.core.types import ASPECTS
from brandsentinel.features.nlp.sentiment import SentimentAnalyzer, get_sentiment_analyzer

log = logging.getLogger(__name__)

# Thứ tự ưu tiên an toàn nghiệp vụ khi hòa (tie-break):
# safety (an toàn tính mạng) > refund (khiếu nại tài chính) > delivery (vận chuyển)
# > quality (chất lượng chung)
ASPECT_SEVERITY_ORDER: dict[str, int] = {
    "safety": 4,
    "refund": 3,
    "delivery": 2,
    "quality": 1,
}

# Regex tách câu theo dấu kết thúc câu và khoảng trắng
SENTENCE_SPLIT_REGEX = re.compile(r"(?<=[.!?\n])\s+")


class AspectExtractor:
    """Trích xuất khía cạnh chủ đạo và phân tích cảm xúc khía cạnh."""

    def __init__(
        self,
        config_path: str | Path | None = None,
        cfg: Config | None = None,
        sentiment_analyzer: SentimentAnalyzer | None = None,
    ) -> None:
        self.cfg = cfg or get_config()
        self.sentiment_analyzer = sentiment_analyzer or get_sentiment_analyzer(self.cfg)

        # Xác định đường dẫn file cấu hình từ điển
        if config_path is not None:
            self.lexicon_path = Path(config_path)
        else:
            repo_root = Path(self.cfg.path("configs")).resolve().parent
            i9_spec = self.cfg.indicators.indicators.get("I9")
            lex_rel = (
                i9_spec.params.get("lexicon", "configs/lexicon_aspect.yaml")
                if i9_spec
                else "configs/lexicon_aspect.yaml"
            )
            self.lexicon_path = repo_root / lex_rel

        self._load_lexicon()

    def _load_lexicon(self) -> None:
        """Đọc và biên dịch các mẫu regex cho từng khía cạnh và ngữ cảnh loại trừ."""
        if not self.lexicon_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file từ điển aspect: {self.lexicon_path}")

        with open(self.lexicon_path, encoding="utf-8") as f:
            data: dict[str, Any] = yaml.safe_load(f) or {}

        # Ngữ cảnh loại trừ (tên sản phẩm, thương hiệu: Fire TV Stick, etc.)
        exclude_list = data.get("exclude_contexts", [])
        self.exclude_patterns: list[re.Pattern] = [
            re.compile(rf"\b{re.escape(str(ctx).lower().strip())}\b", re.IGNORECASE)
            for ctx in exclude_list
            if str(ctx).strip()
        ]

        categories = data.get("categories", {})
        self.aspect_patterns: dict[str, list[re.Pattern]] = {a: [] for a in ASPECTS}

        total_patterns = 0
        for aspect_name in ASPECTS:
            terms = categories.get(aspect_name, [])
            for term in terms:
                term_clean = str(term).lower().strip()
                if term_clean:
                    compiled = re.compile(rf"\b{re.escape(term_clean)}\b", re.IGNORECASE)
                    self.aspect_patterns[aspect_name].append(compiled)
                    total_patterns += 1

        log.info(
            "Đã nạp AspectExtractor từ %s: %d patterns, %d exclude contexts cho %d aspects",
            self.lexicon_path.name,
            total_patterns,
            len(self.exclude_patterns),
            len(ASPECTS),
        )

    def split_sentences(self, text: str) -> list[str]:
        """Tách văn bản thành danh sách câu."""
        if not text:
            return []
        parts = SENTENCE_SPLIT_REGEX.split(text)
        return [p.strip() for p in parts if p.strip()]

    def analyze_sentence_aspects(self, sentence: str) -> dict[str, int]:
        """Đếm số lượt khớp từ khóa của từng aspect trong một câu (có loại trừ ngữ cảnh)."""
        sent_lower = sentence.lower()

        # Vùng loại trừ ngữ cảnh (tên sản phẩm, thương hiệu)
        excluded_spans: list[tuple[int, int]] = []
        for exc_pat in self.exclude_patterns:
            for m in exc_pat.finditer(sent_lower):
                excluded_spans.append((m.start(), m.end()))

        counts: dict[str, int] = {}
        for aspect, patterns in self.aspect_patterns.items():
            match_count = 0
            for pat in patterns:
                for match in pat.finditer(sent_lower):
                    m_start, m_end = match.start(), match.end()
                    # Bỏ qua nếu nằm trong excluded span
                    if any(
                        e_start <= m_start and m_end <= e_end for e_start, e_end in excluded_spans
                    ):
                        continue
                    match_count += 1
            if match_count > 0:
                counts[aspect] = match_count
        return counts

    def find_dominant_aspect(self, text: str | None) -> tuple[str | None, str | None]:
        """Xác định khía cạnh chủ đạo và trích xuất đoạn văn liên quan đến khía cạnh đó.

        Quy tắc Dominant Aspect 3 cấp (Đã được phê duyệt):
          1. Số câu có chứa aspect (n_sentences)
          2. Số lượt khớp từ khóa (n_matches)
          3. Thứ tự an toàn nghiệp vụ (safety > refund > delivery > quality)

        Returns:
            Tuple (dominant_aspect, aspect_text):
            - dominant_aspect: Tên khía cạnh ('quality', 'delivery', 'safety', 'refund') hoặc None.
            - aspect_text: Đoạn văn bản gồm các câu liên quan đến dominant aspect
              (để tính sentiment).
        """
        if text is None or not isinstance(text, str) or not text.strip():
            return None, None

        sentences = self.split_sentences(text)
        if not sentences:
            sentences = [text.strip()]

        aspect_sentences_map: dict[str, list[str]] = {a: [] for a in ASPECTS}
        aspect_matches_map: dict[str, int] = {a: 0 for a in ASPECTS}

        for sent in sentences:
            sent_counts = self.analyze_sentence_aspects(sent)
            for asp, count in sent_counts.items():
                aspect_sentences_map[asp].append(sent)
                aspect_matches_map[asp] += count

        active_aspects = [a for a in ASPECTS if aspect_matches_map[a] > 0]
        if not active_aspects:
            return None, None

        # Sắp xếp theo tuple: (n_sentences, n_matches, severity_priority)
        def score_aspect(asp: str) -> tuple[int, int, int]:
            n_sentences = len(aspect_sentences_map[asp])
            n_matches = aspect_matches_map[asp]
            severity = ASPECT_SEVERITY_ORDER[asp]
            return (n_sentences, n_matches, severity)

        dominant_aspect = max(active_aspects, key=score_aspect)

        # Trích xuất các câu thuộc dominant aspect để phục vụ tính sentiment
        dom_sentences = aspect_sentences_map[dominant_aspect]
        aspect_text = " ".join(dom_sentences) if dom_sentences else text.strip()

        return dominant_aspect, aspect_text

    def extract_aspects(self, clean_reviews: pl.DataFrame) -> pl.DataFrame:
        """Trích xuất aspect và tính aspect_sentiment cho bảng clean_reviews theo batch.

        Args:
            clean_reviews: DataFrame chứa ít nhất [review_id] và [text_norm] (hoặc text_raw).

        Returns:
            DataFrame gồm [review_id, aspect, aspect_sentiment].
        """
        if clean_reviews.is_empty():
            return pl.DataFrame(
                schema={
                    "review_id": pl.String,
                    "aspect": pl.String,
                    "aspect_sentiment": pl.Float32,
                }
            )

        review_ids = clean_reviews["review_id"].to_list()
        if "text_norm" in clean_reviews.columns:
            texts = clean_reviews["text_norm"].to_list()
        elif "text_raw" in clean_reviews.columns:
            texts = clean_reviews["text_raw"].to_list()
        else:
            texts = [None] * len(review_ids)

        dominant_aspects: list[str | None] = []
        aspect_texts_to_score: list[str] = []
        valid_indices: list[int] = []

        # Bước 1: Xác định dominant aspect cho từng review
        for idx, txt in enumerate(texts):
            asp, asp_text = self.find_dominant_aspect(txt)
            dominant_aspects.append(asp)
            if asp is not None and asp_text:
                valid_indices.append(idx)
                aspect_texts_to_score.append(asp_text)

        aspect_sentiments: list[float | None] = [None] * len(review_ids)

        # Bước 2: Tính aspect_sentiment theo batch qua SentimentAnalyzer của Task 2
        if aspect_texts_to_score:
            batch_size = self.cfg.default.nlp.batch_size
            all_scores: list[float] = []

            for start_idx in range(0, len(aspect_texts_to_score), batch_size):
                end_idx = min(start_idx + batch_size, len(aspect_texts_to_score))
                batch_texts = aspect_texts_to_score[start_idx:end_idx]
                p_neg, _, p_pos = self.sentiment_analyzer.analyze_batch(batch_texts)
                scores = np.clip(p_pos - p_neg, -1.0, 1.0)
                all_scores.extend(scores.tolist())

            for i, row_idx in enumerate(valid_indices):
                aspect_sentiments[row_idx] = float(all_scores[i])

        # Đảm bảo luật ngữ nghĩa tuyệt đối: aspect == None -> aspect_sentiment == None
        for i in range(len(review_ids)):
            if dominant_aspects[i] is None:
                aspect_sentiments[i] = None

        return pl.DataFrame(
            {
                "review_id": pl.Series(review_ids, dtype=pl.String),
                "aspect": pl.Series(dominant_aspects, dtype=pl.String),
                "aspect_sentiment": pl.Series(aspect_sentiments, dtype=pl.Float32),
            }
        )


_DEFAULT_EXTRACTOR: AspectExtractor | None = None


def get_aspect_extractor(
    config_path: str | Path | None = None,
    cfg: Config | None = None,
    sentiment_analyzer: SentimentAnalyzer | None = None,
) -> AspectExtractor:
    """Lấy hoặc khởi tạo AspectExtractor singleton."""
    global _DEFAULT_EXTRACTOR
    if _DEFAULT_EXTRACTOR is None or config_path is not None:
        _DEFAULT_EXTRACTOR = AspectExtractor(
            config_path=config_path, cfg=cfg, sentiment_analyzer=sentiment_analyzer
        )
    return _DEFAULT_EXTRACTOR


def compute_aspect(
    clean_reviews: pl.DataFrame,
    cfg: Config | None = None,
    extractor: AspectExtractor | None = None,
) -> pl.DataFrame:
    """Hàm công khai phân tích aspect và aspect_sentiment cho bảng clean_reviews.

    Args:
        clean_reviews: DataFrame clean_reviews.
        cfg: Cấu hình hệ thống (tùy chọn).
        extractor: Thể hiện AspectExtractor tái sử dụng (tùy chọn).

    Returns:
        DataFrame gồm [review_id, aspect, aspect_sentiment].
    """
    active_extractor = extractor or get_aspect_extractor(cfg=cfg)
    return active_extractor.extract_aspects(clean_reviews)


def compute_aspect_window_features(
    reviews_df: pl.DataFrame,
    dates: list[date],
    sku: str,
    category: str,
    *,
    window_days: int = 7,
    window_counts: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Tính các dòng đặc trưng I9 (aspect_neg_*) theo cửa sổ trượt.

    Args:
        reviews_df: DataFrame review của SKU, cần cột date/aspect và sentiment theo aspect.
        dates: Danh sách các mốc ngày liên tục.
        sku: Mã SKU.
        category: Ngành hàng.
        window_days: Kích thước cửa sổ trượt (ngày).
        window_counts: Tham số cũ, giữ để tương thích; n_window được tính theo từng aspect.

    Returns:
        Danh sách dict chứa các dòng đặc trưng I9 cho feature_series.
    """
    del window_counts
    if len(dates) < window_days:
        return []

    # Gom số review và số review tiêu cực cho từng khía cạnh theo ngày
    by_day_aspect_n: dict[tuple[date, str], int] = defaultdict(int)
    by_day_aspect_neg: dict[tuple[date, str], int] = defaultdict(int)

    if not reviews_df.is_empty() and "aspect" in reviews_df.columns:
        cols = ["date", "aspect"]
        has_asp_sent = "aspect_sentiment" in reviews_df.columns
        has_sent = "sentiment_score" in reviews_df.columns
        if has_asp_sent:
            cols.append("aspect_sentiment")
        if has_sent:
            cols.append("sentiment_score")

        for row in reviews_df.select(cols).iter_rows(named=True):
            d = row["date"]
            asp = row["aspect"]
            if asp and asp in ASPECTS:
                by_day_aspect_n[(d, asp)] += 1
                sent = row.get("aspect_sentiment")
                if sent is None:
                    sent = row.get("sentiment_score")
                if sent is not None and sent < 0:
                    by_day_aspect_neg[(d, asp)] += 1

    rows: list[dict[str, Any]] = []

    for index in range(window_days - 1, len(dates)):
        w_end = dates[index]
        curr_dates = dates[index - window_days + 1 : index + 1]

        for asp in ASPECTS:
            total_asp = sum(by_day_aspect_n[(d, asp)] for d in curr_dates)
            neg_asp = sum(by_day_aspect_neg[(d, asp)] for d in curr_dates)
            ratio = float(neg_asp / total_asp) if total_asp > 0 else None

            rows.append(
                {
                    "sku": sku,
                    "category": category,
                    "window_end": w_end,
                    "indicator_id": "I9",
                    "feature": f"aspect_neg_{asp}",
                    "raw_value": ratio,
                    "n_window": int(total_asp),
                }
            )

    return rows
