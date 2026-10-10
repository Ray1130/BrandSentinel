"""Schema các bảng của hợp đồng dữ liệu (docs/data_contract.md) và hàm kiểm tra.

    from brandsentinel.core.schemas import validate, conform, empty_table
    df = validate(Table.CLEAN_REVIEWS, df)      # ném DataContractError nếu vi phạm

Hai tầng kiểm tra:
  1. Pandera: kiểu, null, miền giá trị, khóa duy nhất, cột thừa (strict).
  2. Luật ngữ nghĩa giữa các cột (ví dụ n = n1+...+n5, date = ngày UTC của ts).
Giá trị NaN/inf không hợp lệ: dùng null.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pandera.polars as pa
import polars as pl
from pandera.errors import SchemaError, SchemaErrors

from brandsentinel.core.types import (
    ASPECTS,
    FEATURES_BY_INDICATOR,
    INDICATOR_IDS,
    SCORED_GROUPS,
    Level,
    Table,
)

TS_DTYPE = pl.Datetime("us", "UTC")


class DataContractError(ValueError):
    """Dữ liệu vi phạm hợp đồng."""

    def __init__(self, table: Table, problems: list[str]):
        self.table = table
        self.problems = problems
        super().__init__(
            f"[{table.value}] vi phạm hợp đồng dữ liệu:\n  - " + "\n  - ".join(problems)
        )


@dataclass(frozen=True)
class Col:
    dtype: Any
    nullable: bool = False
    checks: tuple[Any, ...] = ()


_RATING = pa.Check.in_range(1, 5)
_NONNEG = pa.Check.ge(0)
_UNIT = pa.Check.in_range(-1.0, 1.0)
_ZERO_ONE = pa.Check.in_range(0.0, 1.0)

# Khóa duy nhất và cột ngày dùng để phân vùng (core/io.py)
KEYS: dict[Table, list[str]] = {
    Table.CLEAN_REVIEWS: ["review_id"],
    Table.DAILY_AGG: ["sku", "date"],
    Table.NLP_FEATURES: ["review_id"],
    Table.FEATURE_SERIES: ["sku", "window_end", "indicator_id", "feature"],
    Table.INDICATOR_MATRIX: ["sku", "window_end", "indicator_id"],
    Table.ALERTS: ["sku", "window_end"],
    Table.RECALL_LABELS: ["sku", "recall_id"],
}
DATE_COLUMN: dict[Table, str] = {
    Table.CLEAN_REVIEWS: "date",
    Table.DAILY_AGG: "date",
    Table.NLP_FEATURES: "date",
    Table.FEATURE_SERIES: "window_end",
    Table.INDICATOR_MATRIX: "window_end",
    Table.ALERTS: "window_end",
    Table.RECALL_LABELS: "recall_date",
}

COLUMNS: dict[Table, dict[str, Col]] = {
    Table.CLEAN_REVIEWS: {
        "review_id": Col(pl.String),
        "sku": Col(pl.String),
        "category": Col(pl.String),
        "user_id": Col(pl.String, nullable=True),
        "ts": Col(TS_DTYPE),
        "date": Col(pl.Date),
        "rating": Col(pl.Int8, checks=(_RATING,)),
        "text_raw": Col(pl.String, nullable=True),
        "text_norm": Col(pl.String, nullable=True),
        "verified_purchase": Col(pl.Boolean, nullable=True),
        "is_spam": Col(pl.Boolean),
        "spam_reason": Col(pl.String, nullable=True),
    },
    Table.DAILY_AGG: {
        "sku": Col(pl.String),
        "category": Col(pl.String),
        "date": Col(pl.Date),
        **{c: Col(pl.Int32, checks=(_NONNEG,)) for c in ("n", "n_neg")},
        **{f"n{k}": Col(pl.Int32, checks=(_NONNEG,)) for k in range(1, 6)},
        "sum_rating": Col(pl.Float64, checks=(_NONNEG,)),
        "sumsq_rating": Col(pl.Float64, checks=(_NONNEG,)),
        **{c: Col(pl.Int32, checks=(_NONNEG,)) for c in ("n_verified", "n_all", "n_verified_all")},
    },
    Table.NLP_FEATURES: {
        "review_id": Col(pl.String),
        "category": Col(pl.String),
        "date": Col(pl.Date),
        "sentiment_score": Col(pl.Float32, nullable=True, checks=(_UNIT,)),
        "risk_hit": Col(pl.Boolean),
        "risk_terms": Col(pl.List(pl.String), nullable=True),
        "risk_sim": Col(pl.Float32, nullable=True, checks=(_UNIT,)),
        "aspect": Col(pl.String, nullable=True, checks=(pa.Check.isin(list(ASPECTS)),)),
        "aspect_sentiment": Col(pl.Float32, nullable=True, checks=(_UNIT,)),
        "mismatch": Col(pl.Boolean, nullable=True),
    },
    Table.FEATURE_SERIES: {
        "sku": Col(pl.String),
        "category": Col(pl.String),
        "window_end": Col(pl.Date),
        "indicator_id": Col(pl.String, checks=(pa.Check.isin(list(INDICATOR_IDS)),)),
        "feature": Col(pl.String),
        "raw_value": Col(pl.Float64, nullable=True),
        "n_window": Col(pl.Int32, checks=(_NONNEG,)),
    },
    Table.INDICATOR_MATRIX: {
        "sku": Col(pl.String),
        "category": Col(pl.String),
        "window_end": Col(pl.Date),
        "indicator_id": Col(pl.String, checks=(pa.Check.isin(list(INDICATOR_IDS)),)),
        "triggered": Col(pl.Boolean),
        "strength": Col(pl.Float64, nullable=True),
    },
    Table.ALERTS: {
        "sku": Col(pl.String),
        "category": Col(pl.String),
        "window_end": Col(pl.Date),
        "score": Col(pl.Float64, checks=(_ZERO_ONE,)),
        "level": Col(pl.String, checks=(pa.Check.isin([lv.value for lv in Level]),)),
        "groups_active": Col(pl.List(pl.String)),
        "persist": Col(pl.Int8, checks=(_NONNEG,)),
        "verify_flag": Col(pl.Boolean),
        "triggered_ids": Col(pl.List(pl.String)),
    },
    Table.RECALL_LABELS: {
        "sku": Col(pl.String),
        "category": Col(pl.String),
        "recall_id": Col(pl.String),
        "recall_date": Col(pl.Date),
        "source": Col(pl.String),
        "source_url": Col(pl.String),
        "product_name": Col(pl.String),
        "match_type": Col(pl.String, checks=(pa.Check.isin(["exact_asin"]),)),
    },
}

POLARS_SCHEMAS: dict[Table, dict[str, Any]] = {
    t: {name: c.dtype for name, c in cols.items()} for t, cols in COLUMNS.items()
}


def _build(table: Table) -> pa.DataFrameSchema:
    cols = {
        name: pa.Column(c.dtype, checks=list(c.checks) or None, nullable=c.nullable)
        for name, c in COLUMNS[table].items()
    }
    return pa.DataFrameSchema(cols, strict=True, unique=KEYS[table], name=table.value)


SCHEMAS: dict[Table, pa.DataFrameSchema] = {t: _build(t) for t in Table}


# ------------------------- luật ngữ nghĩa giữa các cột -------------------------
def _count_bad(df: pl.DataFrame, ok: pl.Expr) -> int:
    """Số dòng vi phạm; điều kiện null (cột nullable) coi như đạt."""
    return df.filter(~ok.fill_null(True)).height


def _finite(col: str) -> pl.Expr:
    return pl.col(col).is_null() | pl.col(col).is_finite()


def _subset_of(col: str, allowed: list[str]) -> pl.Expr:
    return pl.col(col).list.eval(pl.element().is_in(allowed)).list.all()


def _check_clean_reviews(df: pl.DataFrame) -> list[str]:
    bad = _count_bad(df, pl.col("date") == pl.col("ts").dt.date())
    return [f"{bad} dòng có `date` khác ngày UTC của `ts`"] if bad else []


def _check_daily_agg(df: pl.DataFrame) -> list[str]:
    n_sum = sum(pl.col(f"n{k}") for k in range(1, 6))
    rating_sum = sum(k * pl.col(f"n{k}") for k in range(1, 6))
    sq_sum = sum(k * k * pl.col(f"n{k}") for k in range(1, 6))
    rules = {
        "n != n1+...+n5": pl.col("n") == n_sum,
        "n_neg != n1+n2": pl.col("n_neg") == pl.col("n1") + pl.col("n2"),
        "n_verified > n": pl.col("n_verified") <= pl.col("n"),
        "n > n_all": pl.col("n") <= pl.col("n_all"),
        "n_verified_all > n_all": pl.col("n_verified_all") <= pl.col("n_all"),
        "n_verified > n_verified_all": pl.col("n_verified") <= pl.col("n_verified_all"),
        "sum_rating không khớp histogram": (pl.col("sum_rating") - rating_sum).abs() < 1e-6,
        "sumsq_rating không khớp histogram": (pl.col("sumsq_rating") - sq_sum).abs() < 1e-6,
    }
    return [f"{b} dòng vi phạm: {name}" for name, ok in rules.items() if (b := _count_bad(df, ok))]


def _check_nlp_features(df: pl.DataFrame) -> list[str]:
    rules = {
        "aspect null nhưng aspect_sentiment có giá trị": (
            pl.col("aspect").is_not_null() | pl.col("aspect_sentiment").is_null()
        ),
        "risk_hit = false nhưng risk_terms không rỗng": (
            pl.col("risk_hit")
            | pl.col("risk_terms").is_null()
            | (pl.col("risk_terms").list.len() == 0)
        ),
    }
    return [f"{b} dòng vi phạm: {name}" for name, ok in rules.items() if (b := _count_bad(df, ok))]


def _check_feature_series(df: pl.DataFrame) -> list[str]:
    pairs = [(i, f) for i, fs in FEATURES_BY_INDICATOR.items() for f in fs]
    valid = pl.DataFrame(pairs, schema=["indicator_id", "feature"], orient="row")
    out = []
    unknown = df.join(valid, on=["indicator_id", "feature"], how="anti")
    if unknown.height:
        seen = unknown.select("indicator_id", "feature").unique().head(5).rows()
        out.append(
            f"{unknown.height} dòng có cặp (indicator_id, feature) ngoài hợp đồng, ví dụ {seen}"
        )
    if bad := _count_bad(df, _finite("raw_value")):
        out.append(f"{bad} dòng có raw_value NaN/inf (hãy dùng null)")
    return out


def _check_indicator_matrix(df: pl.DataFrame) -> list[str]:
    bad = _count_bad(df, _finite("strength"))
    return [f"{bad} dòng có strength NaN/inf (hãy dùng null)"] if bad else []


def _check_alerts(df: pl.DataFrame) -> list[str]:
    groups = [g.value for g in SCORED_GROUPS]
    rules = {
        "alerts chỉ được chứa mức MEDIUM hoặc HIGH": pl.col("level").is_in(
            [Level.MEDIUM.value, Level.HIGH.value]
        ),
        "groups_active chứa nhóm ngoài volume/rating/content": _subset_of("groups_active", groups),
        "triggered_ids chứa mã chỉ báo ngoài I1-I11": _subset_of(
            "triggered_ids", list(INDICATOR_IDS)
        ),
    }
    return [f"{b} dòng vi phạm: {name}" for name, ok in rules.items() if (b := _count_bad(df, ok))]


def _check_recall_labels(df: pl.DataFrame) -> list[str]:
    rules = {
        "source_url phải là URL HTTPS": pl.col("source_url").str.starts_with("https://"),
        "source và product_name không được rỗng": (
            (pl.col("source").str.len_chars() > 0) & (pl.col("product_name").str.len_chars() > 0)
        ),
    }
    return [f"{b} dòng vi phạm: {name}" for name, ok in rules.items() if (b := _count_bad(df, ok))]


_SEMANTIC: dict[Table, Callable[[pl.DataFrame], list[str]]] = {
    Table.CLEAN_REVIEWS: _check_clean_reviews,
    Table.DAILY_AGG: _check_daily_agg,
    Table.NLP_FEATURES: _check_nlp_features,
    Table.FEATURE_SERIES: _check_feature_series,
    Table.INDICATOR_MATRIX: _check_indicator_matrix,
    Table.ALERTS: _check_alerts,
    Table.RECALL_LABELS: _check_recall_labels,
}


# ------------------------- API công khai -------------------------
def validate(table: Table, df: pl.DataFrame, *, semantic: bool = True) -> pl.DataFrame:
    """Kiểm tra `df` theo hợp đồng của `table`; trả lại chính `df` nếu hợp lệ."""
    try:
        SCHEMAS[table].validate(df, lazy=True)
    except (SchemaErrors, SchemaError) as e:
        fc = getattr(e, "failure_cases", None)
        detail = str(fc.head(8)) if isinstance(fc, pl.DataFrame) else str(e)[:1500]
        raise DataContractError(table, [f"sai schema:\n{detail}"]) from e
    if semantic:
        problems = _SEMANTIC[table](df)
        if problems:
            raise DataContractError(table, problems)
    return df


def empty_table(table: Table) -> pl.DataFrame:
    """Bảng rỗng đúng kiểu dữ liệu của hợp đồng."""
    return pl.DataFrame(schema=POLARS_SCHEMAS[table])


def conform(table: Table, df: pl.DataFrame) -> pl.DataFrame:
    """Chọn đúng cột theo thứ tự hợp đồng và ép kiểu. Thiếu hoặc thừa cột thì báo lỗi."""
    schema = POLARS_SCHEMAS[table]
    missing = [c for c in schema if c not in df.columns]
    extra = [c for c in df.columns if c not in schema]
    if missing or extra:
        raise DataContractError(table, [f"thiếu cột {missing}", f"thừa cột {extra}"])
    return df.select([pl.col(c).cast(t) for c, t in schema.items()])
