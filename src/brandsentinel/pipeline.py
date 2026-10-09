"""Pipeline stage entry points."""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import date
from pathlib import Path

import polars as pl

from brandsentinel.core.config import Config, get_config, load_config
from brandsentinel.core.io import read_table, write_table
from brandsentinel.core.logging import AuditLog
from brandsentinel.core.types import Table
from brandsentinel.preprocessing.aggregate import make_daily_agg
from brandsentinel.preprocessing.dedup import deduplicate_reviews
from brandsentinel.preprocessing.loader import load_reviews
from brandsentinel.preprocessing.normalize import normalize_reviews
from brandsentinel.preprocessing.spam import filter_eligible_skus, flag_spam_reviews

log = logging.getLogger(__name__)

NlpRunner = Callable[[pl.DataFrame, Config], pl.DataFrame]


def run_preprocess(
    category: str,
    *,
    start: date | None = None,
    end: date | None = None,
    source: str | Path | pl.DataFrame | None = None,
    config: Config | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame, Path | None]:
    """Preprocess one category, persist its tables, and optionally write an audit log."""
    cfg = config or get_config()
    if start and end and start > end:
        raise ValueError("start phải trước hoặc bằng end")
    input_source = source if source is not None else cfg.path("data_raw") / f"{category}.jsonl"
    audit = AuditLog("preprocess") if cfg.default.logging.audit_log else None
    clean = load_reviews(input_source, category)
    clean = normalize_reviews(
        clean,
        audit=audit,
        unicode_form=cfg.default.preprocessing.normalize.unicode_form,
        lowercase=cfg.default.preprocessing.normalize.lowercase,
        strip_html=cfg.default.preprocessing.normalize.strip_html,
        collapse_repeated_chars=cfg.default.preprocessing.normalize.collapse_repeated_chars,
        emoji_to_token=cfg.default.preprocessing.normalize.emoji_to_token,
    )
    clean = deduplicate_reviews(
        clean,
        audit=audit,
        keep=cfg.default.preprocessing.dedup.keep,
    )
    clean = flag_spam_reviews(clean, config=cfg.default.preprocessing.spam, audit=audit)
    daily = make_daily_agg(clean)
    daily = filter_eligible_skus(
        daily,
        selection=cfg.default.data.sku_selection,
        audit=audit,
    )
    selected_skus = daily["sku"].unique().to_list() if not daily.is_empty() else []
    clean = clean.filter(pl.col("sku").is_in(selected_skus))
    if start is not None:
        clean = clean.filter(pl.col("date") >= start)
        daily = daily.filter(pl.col("date") >= start)
    if end is not None:
        clean = clean.filter(pl.col("date") <= end)
        daily = daily.filter(pl.col("date") <= end)
    write_table(clean, Table.CLEAN_REVIEWS, cfg)
    write_table(daily, Table.DAILY_AGG, cfg)
    audit_path = audit.write(cfg.path("data_interim") / "audit") if audit is not None else None
    return clean, daily, audit_path


def _default_nlp_runner(clean: pl.DataFrame, cfg: Config) -> pl.DataFrame:
    """Run NLP feature extraction; requires the optional ``nlp`` dependencies."""
    from brandsentinel.features.nlp.pipeline import compute_nlp_features  # noqa: PLC0415

    return compute_nlp_features(clean, cfg)


def _read_or_none(table: Table, cfg: Config, category: str) -> pl.DataFrame | None:
    try:
        return read_table(table, cfg, category=category)
    except FileNotFoundError:
        return None


def _ensure_nlp(
    clean: pl.DataFrame, cfg: Config, category: str, runner: NlpRunner | None
) -> pl.DataFrame:
    """Chi tinh NLP cho review chua co trong nlp_features, roi tra bang day du."""
    existing = _read_or_none(Table.NLP_FEATURES, cfg, category)
    if existing is None:
        todo = clean
    else:
        todo = clean.filter(~pl.col("review_id").is_in(existing["review_id"]))
    if todo.height:
        have = 0 if existing is None else existing.height
        log.info("NLP: tinh %d review moi (da co %d)", todo.height, have)
        new = (runner or _default_nlp_runner)(todo, cfg)
        write_table(new, Table.NLP_FEATURES, cfg)
        existing = new if existing is None else pl.concat([existing, new], how="diagonal_relaxed")
    assert existing is not None
    # Bo dong cua review da khong con trong clean_reviews (vd dedup doi).
    return existing.filter(pl.col("review_id").is_in(clean["review_id"]))


def run_features(
    category: str,
    *,
    start: date | None = None,
    end: date | None = None,
    config: Config | None = None,
    with_nlp: bool = True,
    nlp_runner: NlpRunner | None = None,
    window_days: int = 7,
) -> pl.DataFrame:
    """Doc clean_reviews/daily_agg -> (NLP) -> feature_series I1-I11 -> ghi qua core/io.

    Tinh feature tren toan bo lich su category; chi ghi cac window_end trong [start, end].
    with_nlp=False chi tao I1-I6 va khong can NLP extra.
    """
    cfg = config or load_config()
    try:
        clean = read_table(Table.CLEAN_REVIEWS, cfg, category=category)
        daily = read_table(Table.DAILY_AGG, cfg, category=category)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Chua co clean_reviews/daily_agg cua '{category}': "
            f"chay `bs run --stage preprocess --category {category}` truoc"
        ) from e

    # Import lazily so volume/rating-only runs do not require the NLP extra.
    from brandsentinel.features import make_feature_series  # noqa: PLC0415

    if with_nlp:
        nlp = _ensure_nlp(clean, cfg, category, nlp_runner)
        clean_f = clean.filter(~pl.col("is_spam").fill_null(False))
        fs = make_feature_series(
            daily_agg=daily, clean_reviews=clean_f, nlp_features=nlp, window_days=window_days
        )
        from brandsentinel.features.reliability import make_reliability_features  # noqa: PLC0415

        rel = make_reliability_features(daily, nlp, clean)
        fs = pl.concat([fs, rel.select(fs.columns)], how="vertical_relaxed")
    else:
        fs = make_feature_series(daily_agg=daily, window_days=window_days)

    if start is not None:
        fs = fs.filter(pl.col("window_end") >= start)
    if end is not None:
        fs = fs.filter(pl.col("window_end") <= end)
    if fs.is_empty():
        log.warning("features: khong co dong nao trong [%s, %s] cho '%s'", start, end, category)
        return fs
    if not with_nlp:
        _refuse_to_drop_content_features(fs, cfg, category)
    write_table(fs, Table.FEATURE_SERIES, cfg)
    return fs


def _refuse_to_drop_content_features(fs: pl.DataFrame, cfg: Config, category: str) -> None:
    """write_table thay the theo SKU/khoang ngay, khong theo indicator_id."""
    old = _read_or_none(Table.FEATURE_SERIES, cfg, category)
    if old is None:
        return
    lo, hi = fs["window_end"].min(), fs["window_end"].max()
    overlap = old.filter(pl.col("sku").is_in(fs["sku"]) & pl.col("window_end").is_between(lo, hi))
    lost = sorted(set(overlap["indicator_id"]) - set(fs["indicator_id"]))
    if lost:
        raise RuntimeError(
            f"run_features: batch chi co {fs['indicator_id'].unique().to_list()} "
            f"nhung da co {lost} trong [{lo}, {hi}] cua '{category}'"
        )


def run_stage(
    stage: str,
    category: str,
    *,
    start: date | None = None,
    end: date | None = None,
    source: str | Path | None = None,
    with_nlp: bool = True,
    config: Config | None = None,
) -> dict[str, object]:
    """Dieu phoi mot stage; tra tom tat de CLI in ra. Them stage moi bang them mot nhanh o day."""
    cfg = config or load_config()
    if stage == "preprocess":
        clean, daily, audit = run_preprocess(
            category, start=start, end=end, source=source, config=cfg
        )
        return {
            "clean_reviews": clean.height,
            "daily_agg": daily.height,
            "audit_log": str(audit) if audit else None,
        }
    if stage == "features":
        fs = run_features(category, start=start, end=end, config=cfg, with_nlp=with_nlp)
        by = fs.group_by("indicator_id").len().sort("indicator_id")
        return {
            "feature_series": fs.height,
            "skus": fs["sku"].n_unique() if fs.height else 0,
            "rows_by_indicator": dict(zip(by["indicator_id"], by["len"], strict=True)),
        }
    raise NotImplementedError(stage)
