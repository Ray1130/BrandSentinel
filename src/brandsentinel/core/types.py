"""Kiểu dùng chung: mức rủi ro, nhóm tín hiệu, tên bảng, mã chỉ báo, tên đặc trưng."""

from __future__ import annotations

from enum import StrEnum


class Level(StrEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"

    @property
    def rank(self) -> int:
        return {"LOW": 0, "MEDIUM": 1, "HIGH": 2}[self.value]

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Level):
            return NotImplemented
        return self.rank < other.rank

    def __le__(self, other: object) -> bool:
        if not isinstance(other, Level):
            return NotImplemented
        return self.rank <= other.rank


class Group(StrEnum):
    VOLUME = "volume"
    RATING = "rating"
    CONTENT = "content"
    FLAG = "flag"


class Table(StrEnum):
    """Tên các bảng trong hợp đồng dữ liệu (docs/data_contract.md)."""

    CLEAN_REVIEWS = "clean_reviews"
    DAILY_AGG = "daily_agg"
    NLP_FEATURES = "nlp_features"
    FEATURE_SERIES = "feature_series"
    INDICATOR_MATRIX = "indicator_matrix"
    ALERTS = "alerts"


INDICATOR_IDS: tuple[str, ...] = tuple(f"I{i}" for i in range(1, 12))
SCORED_GROUPS: tuple[Group, ...] = (Group.VOLUME, Group.RATING, Group.CONTENT)
ASPECTS: tuple[str, ...] = ("quality", "delivery", "safety", "refund")

# Tên đặc trưng (cột `feature` của feature_series) mà mỗi chỉ báo đọc. Khớp `series:` trong
# configs/indicators.yaml. Thêm đặc trưng mới = sửa ở đây + docs/data_contract.md (qua PR).
# Đặc trưng cấp ngày (log1p_daily_count, growth_rate) dùng window_end = chính ngày đó.
FEATURES_BY_INDICATOR: dict[str, tuple[str, ...]] = {
    "I1": ("log1p_daily_count",),
    "I2": ("growth_rate",),
    "I3": ("neg_ratio_shrunk",),
    "I4": ("low_star_ratio_shrunk",),
    "I5": ("rating_rolling_variance",),
    "I6": ("extreme_share", "share_1star", "share_5star"),
    "I7": ("term_burst_score",),
    "I8": ("risk_hits", "risk_distinct_users"),
    "I9": tuple(f"aspect_neg_{a}" for a in ASPECTS),
    "I10": ("mismatch_rate",),
    "I11": ("verified_ratio_all",),
}
