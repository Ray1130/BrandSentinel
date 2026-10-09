"""Risk lexicon detection module for BrandSentinel (I8 - Stage 1).

Phát hiện từ khóa rủi ro an toàn, sức khỏe, pháp lý bằng bộ từ điển,
kết hợp loại trừ ngữ cảnh (exclude context) và phát hiện phủ định (negation window).
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl
import yaml

from brandsentinel.core.config import Config, get_config

log = logging.getLogger(__name__)

# Ký tự ngắt mệnh đề không cho phép phủ định lan truyền
CLAUSE_BREAKERS = {".", "!", "?", ";", "but", "however", "although"}


class RiskLexiconDetector:
    """Bộ phát hiện rủi ro dựa trên từ điển cấu hình trong YAML."""

    def __init__(
        self,
        config_path: str | Path | None = None,
        cfg: Config | None = None,
        *,
        negation_window_tokens: int | None = None,
    ) -> None:
        self.cfg = cfg or get_config()

        # Xác định đường dẫn file cấu hình từ điển
        if config_path is not None:
            self.lexicon_path = Path(config_path)
        else:
            repo_root = Path(self.cfg.path("configs")).resolve().parent
            # Đọc từ indicators.yaml I8 params nếu có
            i8_spec = self.cfg.indicators.indicators.get("I8")
            lex_list = i8_spec.params.get("lexicons", []) if i8_spec else []
            if lex_list:
                self.lexicon_path = repo_root / lex_list[0]
            else:
                self.lexicon_path = repo_root / "configs" / "lexicon_risk_en.yaml"

        self._load_lexicon(negation_window_tokens)

    def _load_lexicon(self, override_negation_window: int | None = None) -> None:
        """Đọc và biên dịch các mẫu regex từ YAML."""
        if not self.lexicon_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file từ điển rủi ro: {self.lexicon_path}")

        with open(self.lexicon_path, encoding="utf-8") as f:
            data: dict[str, Any] = yaml.safe_load(f) or {}

        # Cửa sổ phủ định
        if override_negation_window is not None:
            self.negation_window = override_negation_window
        else:
            self.negation_window = int(data.get("negation_window_tokens", 3))

        # Danh sách từ phủ định
        self.negators = {str(w).lower().strip() for w in data.get("negators", [])}

        # Ngữ cảnh loại trừ (tên sản phẩm, thương hiệu, thành ngữ)
        exclude_list = data.get("exclude_contexts", [])
        self.exclude_patterns: list[re.Pattern] = [
            re.compile(rf"\b{re.escape(str(ctx).lower().strip())}\b", re.IGNORECASE)
            for ctx in exclude_list
            if str(ctx).strip()
        ]

        # Danh mục từ khóa rủi ro
        # Mỗi mục lưu: (compiled_regex, canonical_name, category)
        self.risk_patterns: list[tuple[re.Pattern, str, str]] = []
        categories = data.get("categories", {})
        for cat_name, items in categories.items():
            for item in items:
                pat = str(item.get("pattern", "")).lower().strip()
                canonical = str(item.get("canonical", pat)).strip()
                if pat:
                    compiled = re.compile(rf"\b{re.escape(pat)}\b", re.IGNORECASE)
                    self.risk_patterns.append((compiled, canonical, cat_name))

        log.info(
            "Đã nạp RiskLexiconDetector từ %s: %d risk patterns, %d exclude contexts, "
            "negation_window=%d",
            self.lexicon_path.name,
            len(self.risk_patterns),
            len(self.exclude_patterns),
            self.negation_window,
        )

    def detect_text(self, text: str | None) -> tuple[bool, list[str] | None]:
        """Phát hiện từ khóa rủi ro trong một văn bản.

        Returns:
            Tuple (risk_hit, risk_terms)
            - risk_hit: True nếu có ít nhất 1 match hợp lệ, ngược lại False.
            - risk_terms: List các canonical terms duy nhất, hoặc None nếu không có hit.
        """
        if text is None or not isinstance(text, str) or not text.strip():
            return False, None

        text_lower = text.lower()

        # 1. Tìm các vùng ngữ cảnh loại trừ (exclude spans)
        excluded_spans: list[tuple[int, int]] = []
        for exc_pat in self.exclude_patterns:
            for m in exc_pat.finditer(text_lower):
                excluded_spans.append((m.start(), m.end()))

        # 2. Tokenize để hỗ trợ kiểm tra cửa sổ phủ định
        # Tách token từ và dấu câu
        token_matches = list(re.finditer(r"\b[\w']+\b|[.,!?;]", text_lower))
        tokens = [m.group() for m in token_matches]
        token_starts = [m.start() for m in token_matches]
        valid_canonical_terms: list[str] = []
        term_positions: list[int] = []

        # 3. Quét các mẫu rủi ro
        for pat_regex, canonical, _ in self.risk_patterns:
            for match in pat_regex.finditer(text_lower):
                m_start, m_end = match.start(), match.end()

                # Kiểm tra 1: Có rơi vào ngữ cảnh loại trừ không?
                if any(e_start <= m_start and m_end <= e_end for e_start, e_end in excluded_spans):
                    continue

                # Kiểm tra 2: Có bị phủ định không?
                # Xác định token index của match
                match_token_idx = -1
                for idx, t_start in enumerate(token_starts):
                    if t_start >= m_start:
                        match_token_idx = idx
                        break
                if match_token_idx == -1:
                    match_token_idx = len(tokens)

                # Nhìn ngược lại tối đa negation_window tokens
                is_negated = False
                lookback_start = max(0, match_token_idx - self.negation_window)
                for back_idx in range(match_token_idx - 1, lookback_start - 1, -1):
                    tok = tokens[back_idx]
                    # Nếu gặp dấu ngắt câu hoặc từ nối ngắt mệnh đề -> dừng kiểm tra
                    if tok in CLAUSE_BREAKERS:
                        break
                    # Nếu gặp từ phủ định
                    if tok in self.negators or tok.endswith("n't"):
                        is_negated = True
                        break

                if not is_negated:
                    if canonical not in valid_canonical_terms:
                        valid_canonical_terms.append(canonical)
                        term_positions.append(m_start)

        if valid_canonical_terms:
            # Sắp xếp theo thứ tự xuất hiện đầu tiên trong văn bản
            sorted_terms = [
                term for _, term in sorted(zip(term_positions, valid_canonical_terms, strict=True))
            ]
            return True, sorted_terms

        return False, None

    def compute_risk(self, clean_reviews: pl.DataFrame) -> pl.DataFrame:
        """Tính risk_hit và risk_terms cho toàn bộ DataFrame clean_reviews.

        Args:
            clean_reviews: DataFrame chứa ít nhất [review_id] và [text_norm] (hoặc text_raw).

        Returns:
            DataFrame gồm [review_id, risk_hit, risk_terms].
        """
        if clean_reviews.is_empty():
            return pl.DataFrame(
                schema={
                    "review_id": pl.String,
                    "risk_hit": pl.Boolean,
                    "risk_terms": pl.List(pl.String),
                }
            )

        review_ids = clean_reviews["review_id"].to_list()
        if "text_norm" in clean_reviews.columns:
            texts = clean_reviews["text_norm"].to_list()
        elif "text_raw" in clean_reviews.columns:
            texts = clean_reviews["text_raw"].to_list()
        else:
            texts = [None] * len(review_ids)

        hits: list[bool] = []
        terms_list: list[list[str] | None] = []

        for txt in texts:
            hit, terms = self.detect_text(txt)
            hits.append(hit)
            terms_list.append(terms)

        return pl.DataFrame(
            {
                "review_id": pl.Series(review_ids, dtype=pl.String),
                "risk_hit": pl.Series(hits, dtype=pl.Boolean),
                "risk_terms": pl.Series(terms_list, dtype=pl.List(pl.String)),
            }
        )


_DEFAULT_DETECTOR: RiskLexiconDetector | None = None


def get_risk_detector(
    config_path: str | Path | None = None, cfg: Config | None = None
) -> RiskLexiconDetector:
    """Lấy hoặc khởi tạo RiskLexiconDetector singleton."""
    global _DEFAULT_DETECTOR
    if _DEFAULT_DETECTOR is None or config_path is not None:
        _DEFAULT_DETECTOR = RiskLexiconDetector(config_path=config_path, cfg=cfg)
    return _DEFAULT_DETECTOR


def compute_risk_lexicon(
    clean_reviews: pl.DataFrame,
    cfg: Config | None = None,
    detector: RiskLexiconDetector | None = None,
) -> pl.DataFrame:
    """Hàm công khai phát hiện rủi ro cho bảng clean_reviews.

    Args:
        clean_reviews: DataFrame clean_reviews.
        cfg: Cấu hình hệ thống (tùy chọn).
        detector: Thể hiện detector tái sử dụng (tùy chọn).

    Returns:
        DataFrame gồm [review_id, risk_hit, risk_terms].
    """
    active_detector = detector or get_risk_detector(cfg=cfg)
    return active_detector.compute_risk(clean_reviews)


def compute_risk_window_features(
    reviews_df: pl.DataFrame,
    dates: list[date],
    sku: str,
    category: str,
    *,
    window_days: int = 7,
    window_counts: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Tính các dòng đặc trưng I8 (risk_hits và risk_distinct_users) theo cửa sổ trượt.

    Args:
        reviews_df: DataFrame chứa các review của SKU (cần cột date, user_id, risk_hit).
        dates: Danh sách các mốc ngày liên tục.
        sku: Mã SKU.
        category: Ngành hàng.
        window_days: Kích thước cửa sổ trượt (ngày).
        window_counts: Danh sách tổng review trong từng cửa sổ (n_window).

    Returns:
        Danh sách dict chứa các dòng đặc trưng I8 cho feature_series.
    """
    if len(dates) < window_days:
        return []

    # Gom số hit và tập user theo từng ngày
    by_day_hits: dict[date, int] = defaultdict(int)
    by_day_risk_users: dict[date, set[str]] = defaultdict(set)

    if not reviews_df.is_empty() and "risk_hit" in reviews_df.columns:
        for d, u, hit in reviews_df.select("date", "user_id", "risk_hit").iter_rows():
            if hit:
                by_day_hits[d] += 1
                if u is not None:
                    by_day_risk_users[d].add(u)

    rows: list[dict[str, Any]] = []

    for index in range(window_days - 1, len(dates)):
        w_end = dates[index]
        curr_dates = dates[index - window_days + 1 : index + 1]

        hits_sum = sum(by_day_hits[d] for d in curr_dates)
        distinct_users = len(set().union(*(by_day_risk_users[d] for d in curr_dates)))

        n_window = (
            window_counts[index] if window_counts is not None and index < len(window_counts) else 0
        )

        rows.append(
            {
                "sku": sku,
                "category": category,
                "window_end": w_end,
                "indicator_id": "I8",
                "feature": "risk_hits",
                "raw_value": float(hits_sum),
                "n_window": int(n_window),
            }
        )
        rows.append(
            {
                "sku": sku,
                "category": category,
                "window_end": w_end,
                "indicator_id": "I8",
                "feature": "risk_distinct_users",
                "raw_value": float(distinct_users),
                "n_window": int(n_window),
            }
        )

    return rows
