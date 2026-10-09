"""Feature generation from validated preprocessing tables."""

from __future__ import annotations

import polars as pl

from brandsentinel.core.schemas import empty_table, validate
from brandsentinel.core.types import Table
from brandsentinel.features.rating import make_rating_features
from brandsentinel.features.volume import make_volume_features
from brandsentinel.features.windows import validate_window_days
from brandsentinel.preprocessing.validate import validate_daily_agg

__all__ = [
    "make_content_features",
    "make_feature_series",
    "make_rating_features",
    "make_volume_features",
]


def make_feature_series(
    daily_agg: pl.DataFrame | None = None,
    clean_reviews: pl.DataFrame | None = None,
    nlp_features: pl.DataFrame | None = None,
    *,
    window_days: int = 7,
) -> pl.DataFrame:
    """Tạo và kiểm thực feature_series cho I1-I9.

    Thứ tự đối số chính thức là ``(daily_agg, clean_reviews, nlp_features)``.
    Dạng vị trí ``(clean_reviews, daily_agg, nlp_features)`` cũ vẫn được
    nhận diện; ưu tiên dùng tên đối số để tránh nhập nhằng.

    Ví dụ:
    - ``make_feature_series(daily_agg=daily)`` tạo I1-I6.
    - ``make_feature_series(
          daily_agg=daily,
          clean_reviews=clean,
          nlp_features=nlp,
          window_days=7,
      )`` tạo I1-I9.
    - ``make_feature_series(
          clean_reviews=clean,
          nlp_features=nlp,
          window_days=7,
      )`` tạo I7-I9.
    """
    validate_window_days(window_days)

    # Phân giải linh hoạt khi truyền vị trí (clean_reviews, daily_agg, nlp_features).
    if daily_agg is not None and "user_id" in daily_agg.columns:
        actual_clean = daily_agg
        actual_daily = clean_reviews
        actual_nlp = nlp_features
    else:
        actual_daily = daily_agg
        actual_clean = clean_reviews
        actual_nlp = nlp_features

    feature_frames: list[pl.DataFrame] = []

    # Đặc trưng volume và rating (I1-I6).
    if actual_daily is not None:
        validate_daily_agg(actual_daily)

        if not actual_daily.is_empty():
            feature_frames.append(make_volume_features(actual_daily, window_days=window_days))
            feature_frames.append(make_rating_features(actual_daily, window_days=window_days))

    # Đặc trưng nội dung và ngữ nghĩa (I7-I9).
    if actual_clean is not None and actual_nlp is not None:
        if not actual_clean.is_empty() and not actual_nlp.is_empty():
            # Lazy import: chỉ nạp module content khi thực sự cần NLP features.
            from brandsentinel.features.content import make_content_features

            content_df = make_content_features(
                actual_clean,
                actual_nlp,
                daily_agg=actual_daily,
                window_days=window_days,
            )
            feature_frames.append(content_df)

    if not feature_frames:
        return validate(
            Table.FEATURE_SERIES,
            empty_table(Table.FEATURE_SERIES),
        )

    result = pl.concat(feature_frames)
    return validate(Table.FEATURE_SERIES, result)


def __getattr__(name: str):
    """Nạp make_content_features khi được truy cập trực tiếp."""
    if name == "make_content_features":
        from brandsentinel.features.content import make_content_features

        return make_content_features

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
