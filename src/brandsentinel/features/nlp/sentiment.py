"""Sentiment analysis and mismatch detection module for BrandSentinel.

Phân tích cảm xúc văn bản review và phát hiện bất nhất giữa rating và cảm xúc.
Mô hình mặc định: cardiffnlp/twitter-roberta-base-sentiment-latest
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from brandsentinel.core.config import Config, get_config

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)


class SentimentAnalyzer:
    """Quản lý mô hình phân tích cảm xúc và suy luận theo batch."""

    def __init__(
        self,
        cfg: Config | None = None,
        *,
        model_name: str | None = None,
        device: str | None = None,
        batch_size: int | None = None,
        max_length: int = 128,
    ) -> None:
        self.cfg = cfg or get_config()

        # Đọc tham số từ config
        nlp_cfg = self.cfg.default.nlp
        self.model_name = (
            model_name
            or nlp_cfg.sentiment_model.get("en", "cardiffnlp/twitter-roberta-base-sentiment-latest")
        )
        self.batch_size = batch_size or nlp_cfg.batch_size
        self.max_length = max_length

        # Xác định thiết bị tính toán
        target_device = device or nlp_cfg.device
        if target_device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = target_device

        # Tham số phát hiện bất nhất (I10) từ indicators.yaml
        i10_spec = self.cfg.indicators.indicators.get("I10")
        i10_params = i10_spec.params if i10_spec else {}
        self.text_negative_min = float(i10_params.get("text_negative_min", 0.80))
        self.rating_high_min = int(i10_params.get("rating_high_min", 4))
        self.text_positive_min = float(i10_params.get("text_positive_min", 0.80))
        self.rating_low_max = int(i10_params.get("rating_low_max", 2))

        log.info(
            "Khởi tạo SentimentAnalyzer: model=%s, device=%s, batch_size=%d",
            self.model_name,
            self.device,
            self.batch_size,
        )

        # Nạp tokenizer và model (chỉ nạp một lần)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.eval()
        if self.device == "cuda":
            self.model.to("cuda")

        # Ánh xạ nhãn model
        self.id2label = {int(k): str(v).lower() for k, v in self.model.config.id2label.items()}
        self._pos_idx = self._find_label_index("pos")
        self._neg_idx = self._find_label_index("neg")
        self._neu_idx = self._find_label_index("neu")

    def _find_label_index(self, pattern: str) -> int:
        for idx, name in self.id2label.items():
            if pattern in name:
                return idx
        # Fallback theo quy ước thông dụng của Cardiff NLP (0: neg, 1: neu, 2: pos)
        fallback = {"neg": 0, "neu": 1, "pos": 2}
        return fallback.get(pattern, 0)

    def analyze_batch(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Suy luận phân phối xác suất cho một batch văn bản.

        Returns:
            Tuple (p_neg, p_neu, p_pos) dưới dạng mảng 1D float32.
        """
        if not texts:
            empty = np.array([], dtype=np.float32)
            return empty, empty, empty

        with torch.no_grad():
            encoded = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            if self.device == "cuda":
                encoded = {k: v.to("cuda") for k, v in encoded.items()}

            outputs = self.model(**encoded)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()

        p_neg = probs[:, self._neg_idx].astype(np.float32)
        p_neu = probs[:, self._neu_idx].astype(np.float32)
        p_pos = probs[:, self._pos_idx].astype(np.float32)
        return p_neg, p_neu, p_pos

    def compute_sentiment(self, clean_reviews: pl.DataFrame) -> pl.DataFrame:
        """Tính sentiment_score và mismatch cho DataFrame clean_reviews.

        Args:
            clean_reviews: DataFrame chứa ít nhất [review_id, rating, text_norm] (hoặc text_raw).

        Returns:
            DataFrame chứa [review_id, sentiment_score, mismatch].
        """
        if clean_reviews.is_empty():
            return pl.DataFrame(
                schema={
                    "review_id": pl.String,
                    "sentiment_score": pl.Float32,
                    "mismatch": pl.Boolean,
                }
            )

        n_rows = clean_reviews.height
        review_ids = clean_reviews["review_id"].to_list()
        ratings = clean_reviews["rating"].to_list()

        # Lấy văn bản từ text_norm (fallback text_raw nếu không có)
        if "text_norm" in clean_reviews.columns:
            raw_texts = clean_reviews["text_norm"].to_list()
        elif "text_raw" in clean_reviews.columns:
            raw_texts = clean_reviews["text_raw"].to_list()
        else:
            raw_texts = [None] * n_rows

        # Xác định các dòng có text hợp lệ
        valid_indices: list[int] = []
        valid_texts: list[str] = []

        for idx, txt in enumerate(raw_texts):
            if txt is not None and isinstance(txt, str) and txt.strip():
                valid_indices.append(idx)
                valid_texts.append(txt.strip())

        # Khởi tạo mảng kết quả với None
        out_sentiment_scores: list[float | None] = [None] * n_rows
        out_mismatch: list[bool | None] = [None] * n_rows

        # Xử lý các dòng có text hợp lệ theo batch
        total_valid = len(valid_texts)
        if total_valid > 0:
            all_p_neg: list[float] = []
            all_p_pos: list[float] = []

            for start_idx in range(0, total_valid, self.batch_size):
                end_idx = min(start_idx + self.batch_size, total_valid)
                batch_texts = valid_texts[start_idx:end_idx]
                p_neg, _, p_pos = self.analyze_batch(batch_texts)
                all_p_neg.extend(p_neg.tolist())
                all_p_pos.extend(p_pos.tolist())

            for i, row_idx in enumerate(valid_indices):
                neg_prob = all_p_neg[i]
                pos_prob = all_p_pos[i]
                rating = ratings[row_idx]

                # Production mapping: sentiment_score = P_positive - P_negative
                score = float(np.clip(pos_prob - neg_prob, -1.0, 1.0))
                out_sentiment_scores[row_idx] = score

                # Logic mismatch (I10) theo indicators.yaml:
                if rating == 3:
                    mismatch = False
                elif rating >= self.rating_high_min and neg_prob >= self.text_negative_min:
                    mismatch = True
                elif rating <= self.rating_low_max and pos_prob >= self.text_positive_min:
                    mismatch = True
                else:
                    mismatch = False

                out_mismatch[row_idx] = mismatch

        # Với các dòng text rỗng / null: nếu có rating thì mismatch = False (hoặc None theo chuẩn)
        for idx in range(n_rows):
            if idx not in valid_indices:
                out_sentiment_scores[idx] = None
                out_mismatch[idx] = None

        return pl.DataFrame(
            {
                "review_id": pl.Series(review_ids, dtype=pl.String),
                "sentiment_score": pl.Series(out_sentiment_scores, dtype=pl.Float32),
                "mismatch": pl.Series(out_mismatch, dtype=pl.Boolean),
            }
        )


# Singleton instance để tái sử dụng
_DEFAULT_ANALYZER: SentimentAnalyzer | None = None


def get_sentiment_analyzer(cfg: Config | None = None) -> SentimentAnalyzer:
    """Lấy hoặc khởi tạo SentimentAnalyzer singleton để tránh nạp lại mô hình."""
    global _DEFAULT_ANALYZER
    if _DEFAULT_ANALYZER is None:
        _DEFAULT_ANALYZER = SentimentAnalyzer(cfg=cfg)
    return _DEFAULT_ANALYZER


def compute_sentiment(
    clean_reviews: pl.DataFrame,
    cfg: Config | None = None,
    analyzer: SentimentAnalyzer | None = None,
) -> pl.DataFrame:
    """Hàm công khai tính toán sentiment và mismatch cho bảng clean_reviews.

    Args:
        clean_reviews: DataFrame chứa dữ liệu clean_reviews.
        cfg: Cấu hình hệ thống (tùy chọn).
        analyzer: Thể hiện SentimentAnalyzer tái sử dụng (tùy chọn).

    Returns:
        DataFrame gồm các cột [review_id, sentiment_score, mismatch].
    """
    active_analyzer = analyzer or get_sentiment_analyzer(cfg=cfg)
    return active_analyzer.compute_sentiment(clean_reviews)
