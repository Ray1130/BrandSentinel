"""Topic burst detection module for BrandSentinel (Indicator I7).

Xác định đột biến tần suất cụm từ (term burst) theo cửa sổ trượt thời gian,
phát hiện sớm các chủ đề sự cố mới nổi dựa trên n-gram và số người dùng phân biệt.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter, defaultdict
from datetime import date
from typing import Any

import polars as pl

from brandsentinel.core.config import Config, get_config

log = logging.getLogger(__name__)

# Bộ stop words tiếng Anh tiêu chuẩn, gọn nhẹ, chạy nhanh không phụ thuộc NLTK/SpaCy
ENGLISH_STOP_WORDS: frozenset[str] = frozenset(
    {
        "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
        "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
        "but", "by", "can", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't",
        "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't",
        "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
        "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll",
        "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "just",
        "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off",
        "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over",
        "own", "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so",
        "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
        "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've",
        "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't",
        "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when",
        "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's",
        "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your",
        "yours", "yourself", "yourselves",
    }
)

TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def extract_ngrams(
    text: str | None,
    ngram_range: tuple[int, int] = (1, 2),
    *,
    min_token_len: int = 3,
    stop_words: frozenset[str] = ENGLISH_STOP_WORDS,
) -> list[str]:
    """Trích xuất unigrams và bigrams từ văn bản đã chuẩn hóa.

    Args:
        text: Chuỗi văn bản review.
        ngram_range: Khoảng n-gram (mặc định (1, 2)).
        min_token_len: Độ dài tối thiểu của một từ hợp lệ.
        stop_words: Tập các từ dừng cần loại bỏ.

    Returns:
        Danh sách các n-gram hợp lệ.
    """
    if not text or not isinstance(text, str):
        return []

    tokens = [
        tok
        for tok in TOKEN_PATTERN.findall(text.lower())
        if len(tok) >= min_token_len and tok not in stop_words
    ]
    if not tokens:
        return []

    min_n, max_n = ngram_range
    results: list[str] = []

    # Unigrams
    if min_n <= 1 <= max_n:
        results.extend(tokens)

    # Bigrams
    if max_n >= 2 and len(tokens) >= 2:
        for i in range(len(tokens) - 1):
            results.append(f"{tokens[i]} {tokens[i + 1]}")

    return results


def compute_term_burst(
    texts: list[str | None],
    user_ids: list[str],
    prior_texts: list[str | None] | None = None,
    *,
    ngram_range: tuple[int, int] = (1, 2),
    min_term_count: int = 5,
    min_distinct_users: int = 3,
) -> tuple[float, str | None]:
    """Tính điểm bùng nổ cụm từ (term burst score) cho một tập review trong cửa sổ.

    Args:
        texts: Danh sách văn bản review trong cửa sổ hiện tại.
        user_ids: Danh sách user_id tương ứng với từng review.
        prior_texts: Danh sách văn bản review trong cửa sổ trước (để so sánh baseline).
        ngram_range: Khoảng n-gram cần trích xuất.
        min_term_count: Ngưỡng xuất hiện tối thiểu của cụm từ trong cửa sổ hiện tại.
        min_distinct_users: Số user phân biệt tối thiểu sử dụng cụm từ đó.

    Returns:
        Tuple (burst_score, top_term). Nếu không có từ nào đạt ngưỡng, trả về (0.0, None).
    """
    if not texts:
        return 0.0, None

    # Đếm tần suất và user_id cho cửa sổ hiện tại
    curr_counts: Counter[str] = Counter()
    curr_users: dict[str, set[str]] = defaultdict(set)

    for text, uid in zip(texts, user_ids):
        terms = extract_ngrams(text, ngram_range=ngram_range)
        for term in terms:
            curr_counts[term] += 1
            curr_users[term].add(uid)

    # Lọc các ứng viên thỏa mãn điều kiện min_term_count và min_distinct_users
    candidates = [
        term
        for term, count in curr_counts.items()
        if count >= min_term_count and len(curr_users[term]) >= min_distinct_users
    ]

    if not candidates:
        return 0.0, None

    # Đếm tần suất trong cửa sổ trước (nếu có)
    prior_counts: Counter[str] = Counter()
    if prior_texts:
        for text in prior_texts:
            for term in extract_ngrams(text, ngram_range=ngram_range):
                prior_counts[term] += 1

    # Tính Poisson residual burst score: (c_curr - c_prior) / sqrt(c_prior + 1)
    best_term: str | None = None
    max_score = 0.0

    for term in candidates:
        c_curr = curr_counts[term]
        c_prior = prior_counts.get(term, 0)
        # Nếu từ mới xuất hiện hoàn toàn: c_prior = 0 -> burst = c_curr / 1.0
        score = (c_curr - c_prior) / math.sqrt(c_prior + 1.0)
        if score > max_score:
            max_score = score
            best_term = term

    return float(max(0.0, max_score)), best_term


def compute_topic_window_features(
    reviews_df: pl.DataFrame,
    dates: list[date],
    sku: str,
    category: str,
    *,
    window_days: int = 7,
    min_term_count: int = 5,
    min_distinct_users: int = 3,
    window_counts: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Tính các dòng đặc trưng term_burst_score cho một SKU qua các cửa sổ trượt.

    Args:
        reviews_df: DataFrame các review của SKU (cần có cột date, user_id, text_norm/text_raw).
        dates: Danh sách ngày liên tục của chuỗi thời gian.
        sku: Mã SKU.
        category: Ngành hàng.
        window_days: Kích thước cửa sổ trượt (ngày).
        min_term_count: Số lần xuất hiện tối thiểu.
        min_distinct_users: Số user phân biệt tối thiểu.
        window_counts: Danh sách tổng số review trong từng cửa sổ (n_window).

    Returns:
        Danh sách dict chứa các dòng đặc trưng I7 (term_burst_score).
    """
    if len(dates) < window_days:
        return []

    # Gom review theo ngày để truy xuất nhanh O(1)
    by_day_texts: dict[date, list[str | None]] = defaultdict(list)
    by_day_users: dict[date, list[str]] = defaultdict(list)

    if not reviews_df.is_empty():
        text_col = "text_norm" if "text_norm" in reviews_df.columns else "text_raw"
        for d, u, t in reviews_df.select("date", "user_id", text_col).iter_rows():
            by_day_texts[d].append(t)
            by_day_users[d].append(u)

    rows: list[dict[str, Any]] = []

    for index in range(window_days - 1, len(dates)):
        w_end = dates[index]
        curr_dates = dates[index - window_days + 1 : index + 1]

        # Cửa sổ quá khứ liền kề
        prior_start_idx = max(0, index - 2 * window_days + 1)
        prior_end_idx = index - window_days + 1
        prior_dates = dates[prior_start_idx:prior_end_idx] if index >= window_days else []

        # Gom văn bản và user
        curr_texts: list[str | None] = []
        curr_uids: list[str] = []
        for d in curr_dates:
            curr_texts.extend(by_day_texts[d])
            curr_uids.extend(by_day_users[d])

        prior_texts: list[str | None] = []
        for d in prior_dates:
            prior_texts.extend(by_day_texts[d])

        n_window = (
            window_counts[index]
            if window_counts is not None and index < len(window_counts)
            else len(curr_texts)
        )

        if not curr_texts:
            score: float | None = None
        else:
            burst_score, _ = compute_term_burst(
                curr_texts,
                curr_uids,
                prior_texts,
                min_term_count=min_term_count,
                min_distinct_users=min_distinct_users,
            )
            score = burst_score

        rows.append(
            {
                "sku": sku,
                "category": category,
                "window_end": w_end,
                "indicator_id": "I7",
                "feature": "term_burst_score",
                "raw_value": score,
                "n_window": int(n_window),
            }
        )

    return rows
