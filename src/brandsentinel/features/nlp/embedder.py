"""Semantic embedding and risk similarity module for BrandSentinel (I8 - Layer 2).

Thực hiện nhúng ngữ nghĩa (sentence embeddings) cho nội dung review và tính toán
độ tương đồng ngữ nghĩa (cosine similarity) với các câu mẫu/khái niệm rủi ro (risk prototypes)
cho 4 danh mục rủi ro của I8: safety, health, counterfeit, legal.

Mô hình mặc định: sentence-transformers/all-MiniLM-L6-v2 (kích thước 384, nhanh trên CPU).
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
from sentence_transformers import SentenceTransformer

from brandsentinel.core.config import Config, get_config
from brandsentinel.features.nlp.risk_lexicon import (
    RiskLexiconDetector,
    get_risk_detector,
)

log = logging.getLogger(__name__)

# Câu mẫu đại diện cho 4 danh mục rủi ro theo taxonomy của lexicon_risk_en.yaml
DEFAULT_RISK_PROTOTYPES: dict[str, str] = {
    "safety": "Product caught fire, overheated, exploded, caused electric shock or burning smell fire hazard.",
    "health": "Product caused allergic reaction, severe skin rash, injury, toxic chemical burn or choking hazard.",
    "counterfeit": "Product is fake, counterfeit, knockoff replica or unauthorized copy.",
    "legal": "Product was recalled by manufacturer or subject to legal lawsuit class action.",
}


class RiskEmbedder:
    """Quản lý mô hình nhúng và tính toán độ tương đồng rủi ro ngữ nghĩa."""

    def __init__(
        self,
        cfg: Config | None = None,
        *,
        model_name: str | None = None,
        device: str | None = None,
        batch_size: int | None = None,
        prototypes: dict[str, str] | None = None,
        similarity_threshold: float | None = None,
    ) -> None:
        self.cfg = cfg or get_config()

        # Đọc tham số NLP từ config
        nlp_cfg = self.cfg.default.nlp
        self.model_name = model_name or nlp_cfg.embedding_model
        self.batch_size = batch_size or nlp_cfg.batch_size

        # Thiết bị tính toán
        target_device = device or nlp_cfg.device
        if target_device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = target_device

        # Tham số I8 từ indicators.yaml
        i8_spec = self.cfg.indicators.indicators.get("I8")
        i8_params = i8_spec.params if i8_spec else {}
        if similarity_threshold is not None:
            self.similarity_threshold = float(similarity_threshold)
        else:
            self.similarity_threshold = float(i8_params.get("embedding_similarity_min", 0.60))

        # Cấu hình prototypes
        self.prototypes = prototypes or dict(DEFAULT_RISK_PROTOTYPES)
        self.proto_categories = list(self.prototypes.keys())
        self.proto_texts = [self.prototypes[cat] for cat in self.proto_categories]

        log.info(
            "Khởi tạo RiskEmbedder: model=%s, device=%s, batch_size=%d, threshold=%.2f",
            self.model_name,
            self.device,
            self.batch_size,
            self.similarity_threshold,
        )

        # Nạp mô hình một lần duy nhất
        self.model = SentenceTransformer(self.model_name, device=self.device)
        if hasattr(self.model, "get_embedding_dimension"):
            self.dimension = self.model.get_embedding_dimension()
        else:
            self.dimension = self.model.get_sentence_embedding_dimension()

        # Tiền tính toán embedding cho prototypes (đã chuẩn hóa L2)
        self.proto_embeddings: np.ndarray = self.model.encode(
            self.proto_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def encode(
        self,
        texts: list[str | None] | Iterable[str | None],
        batch_size: int | None = None,
    ) -> np.ndarray:
        """Mã hóa danh sách văn bản thành ma trận embedding chuẩn hóa L2.

        Văn bản rỗng hoặc None được biểu diễn bằng vector 0 có kích thước self.dimension.

        Args:
            texts: Danh sách chuỗi văn bản.
            batch_size: Kích thước batch (mặc định dùng self.batch_size).

        Returns:
            np.ndarray có shape (N, dimension) kiểu float32.
        """
        raw_list = list(texts)
        n = len(raw_list)
        if n == 0:
            return np.empty((0, self.dimension), dtype=np.float32)

        # Tách các văn bản hợp lệ để encode
        valid_indices: list[int] = []
        valid_texts: list[str] = []
        for idx, t in enumerate(raw_list):
            if t is not None and isinstance(t, str) and t.strip():
                valid_indices.append(idx)
                valid_texts.append(t.strip())

        output_embeddings = np.zeros((n, self.dimension), dtype=np.float32)

        if valid_texts:
            bs = batch_size or self.batch_size
            encoded = self.model.encode(
                valid_texts,
                batch_size=bs,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            for orig_idx, emb in zip(valid_indices, encoded, strict=True):
                output_embeddings[orig_idx] = emb

        return output_embeddings

    def similarity_to_prototypes(
        self,
        texts: list[str | None] | Iterable[str | None],
    ) -> tuple[list[float | None], dict[str, list[float | None]]]:
        """Tính toán độ tương đồng cosine giữa các văn bản và risk prototypes.

        Returns:
            Tuple gồm:
            - max_sim: danh sách giá trị tương đồng lớn nhất trong [-1.0, 1.0], None nếu văn bản rỗng/null.
            - category_sims: dict ánh xạ từng category sang danh sách điểm tương đồng tương ứng.
        """
        raw_list = list(texts)
        n = len(raw_list)
        if n == 0:
            return [], {cat: [] for cat in self.proto_categories}

        valid_indices: list[int] = []
        valid_texts: list[str] = []
        for idx, t in enumerate(raw_list):
            if t is not None and isinstance(t, str) and t.strip():
                valid_indices.append(idx)
                valid_texts.append(t.strip())

        max_sims: list[float | None] = [None] * n
        category_sims: dict[str, list[float | None]] = {cat: [None] * n for cat in self.proto_categories}

        if valid_texts:
            # Mã hóa văn bản hợp lệ
            text_emb = self.model.encode(
                valid_texts,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            # Cosine similarity: ma trận (len(valid), num_prototypes) = text_emb @ proto_embeddings.T
            sim_matrix = np.dot(text_emb, self.proto_embeddings.T)

            # Cắt giá trị trong khoảng [-1.0, 1.0] để tránh sai số số học
            sim_matrix = np.clip(sim_matrix, -1.0, 1.0)

            valid_max = np.max(sim_matrix, axis=1)

            for i, orig_idx in enumerate(valid_indices):
                max_sims[orig_idx] = float(valid_max[i])
                for c_idx, cat in enumerate(self.proto_categories):
                    category_sims[cat][orig_idx] = float(sim_matrix[i, c_idx])

        return max_sims, category_sims

    def compute_risk_sim(self, clean_reviews: pl.DataFrame) -> pl.DataFrame:
        """Tính cột risk_sim cho bảng clean_reviews.

        Args:
            clean_reviews: DataFrame chứa ít nhất [review_id] và [text_norm] (hoặc text_raw).

        Returns:
            DataFrame gồm [review_id, risk_sim] với risk_sim kiểu Float32 trong [-1.0, 1.0].
        """
        if clean_reviews.is_empty():
            return pl.DataFrame(
                schema={
                    "review_id": pl.String,
                    "risk_sim": pl.Float32,
                }
            )

        review_ids = clean_reviews["review_id"].to_list()
        if "text_norm" in clean_reviews.columns:
            texts = clean_reviews["text_norm"].to_list()
        elif "text_raw" in clean_reviews.columns:
            texts = clean_reviews["text_raw"].to_list()
        else:
            texts = [None] * len(review_ids)

        max_sims, _ = self.similarity_to_prototypes(texts)

        return pl.DataFrame(
            {
                "review_id": pl.Series(review_ids, dtype=pl.String),
                "risk_sim": pl.Series(max_sims, dtype=pl.Float32),
            }
        )


_DEFAULT_EMBEDDER: RiskEmbedder | None = None


def get_risk_embedder(
    cfg: Config | None = None,
    *,
    model_name: str | None = None,
    device: str | None = None,
    similarity_threshold: float | None = None,
) -> RiskEmbedder:
    """Lấy hoặc khởi tạo RiskEmbedder singleton."""
    global _DEFAULT_EMBEDDER
    if _DEFAULT_EMBEDDER is None or model_name is not None:
        _DEFAULT_EMBEDDER = RiskEmbedder(
            cfg=cfg,
            model_name=model_name,
            device=device,
            similarity_threshold=similarity_threshold,
        )
    return _DEFAULT_EMBEDDER


def compute_embedding_similarity(
    clean_reviews: pl.DataFrame,
    cfg: Config | None = None,
    embedder: RiskEmbedder | None = None,
) -> pl.DataFrame:
    """Hàm công khai tính toán risk_sim cho clean_reviews.

    Returns:
        DataFrame gồm [review_id, risk_sim].
    """
    active_embedder = embedder or get_risk_embedder(cfg=cfg)
    return active_embedder.compute_risk_sim(clean_reviews)


def compute_i8_combined(
    clean_reviews: pl.DataFrame,
    cfg: Config | None = None,
    *,
    detector: RiskLexiconDetector | None = None,
    embedder: RiskEmbedder | None = None,
    threshold: float | None = None,
    mode: str = "combined",
) -> pl.DataFrame:
    """Tính toán tín hiệu I8 hoàn chỉnh kết hợp Layer 1 (Lexicon) và Layer 2 (Embedding).

    Args:
        clean_reviews: DataFrame chứa [review_id, text_norm / text_raw].
        cfg: Cấu hình hệ thống.
        detector: RiskLexiconDetector (tùy chọn).
        embedder: RiskEmbedder (tùy chọn).
        threshold: Ngưỡng tương đồng embedding (nếu None, lấy từ config).
        mode:
            - 'layer1': chỉ dùng Layer 1 (rule-based).
            - 'layer2': chỉ dùng Layer 2 (semantic embedding).
            - 'combined': kết hợp Layer 1 OR (risk_sim >= threshold).

    Returns:
        DataFrame gồm [review_id, risk_hit, risk_terms, risk_sim] tuân thủ schema NLP_FEATURES.
    """
    if clean_reviews.is_empty():
        return pl.DataFrame(
            schema={
                "review_id": pl.String,
                "risk_hit": pl.Boolean,
                "risk_terms": pl.List(pl.String),
                "risk_sim": pl.Float32,
            }
        )

    active_detector = detector or get_risk_detector(cfg=cfg)
    active_embedder = embedder or get_risk_embedder(cfg=cfg)

    # 1. Chạy Layer 1: Rule-based lexicon
    l1_df = active_detector.compute_risk(clean_reviews)

    # 2. Chạy Layer 2: Semantic embedding similarity
    l2_df = active_embedder.compute_risk_sim(clean_reviews)

    # Join theo review_id
    joined = l1_df.join(l2_df, on="review_id", how="left")

    th = threshold if threshold is not None else active_embedder.similarity_threshold

    if mode == "layer1":
        # Giữ nguyên risk_hit từ Layer 1
        return joined.select(["review_id", "risk_hit", "risk_terms", "risk_sim"])

    if mode == "layer2":
        # Chỉ dựa trên risk_sim >= th
        sim_hit_expr = pl.col("risk_sim").is_not_null() & (pl.col("risk_sim") >= th)
        return joined.with_columns(
            sim_hit_expr.alias("risk_hit"),
            pl.lit(None, dtype=pl.List(pl.String)).alias("risk_terms"),
        ).select(["review_id", "risk_hit", "risk_terms", "risk_sim"])

    # mode == "combined"
    # Tín hiệu kết hợp: Layer 1 HIT HOẶC (risk_sim >= th)
    combined_hit_expr = pl.col("risk_hit") | (
        pl.col("risk_sim").is_not_null() & (pl.col("risk_sim") >= th)
    )
    result = joined.with_columns(combined_hit_expr.alias("risk_hit"))

    # Đảm bảo luật dữ liệu: Nếu risk_hit == False thì risk_terms bắt buộc là null
    result = result.with_columns(
        pl.when(pl.col("risk_hit"))
        .then(pl.col("risk_terms"))
        .otherwise(pl.lit(None, dtype=pl.List(pl.String)))
        .alias("risk_terms")
    )

    return result.select(["review_id", "risk_hit", "risk_terms", "risk_sim"])
