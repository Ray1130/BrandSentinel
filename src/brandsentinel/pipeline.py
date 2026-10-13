"""Pipeline stage entry points."""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date
from pathlib import Path
from typing import Any

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
        daily = read_table(Table.DAILY_AGG, cfg, category=category)
        clean = read_table(Table.CLEAN_REVIEWS, cfg, category=category) if with_nlp else None
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
        # Streaming chunking by SKU to guarantee RAM < 350MB even for large categories like Home_and_Kitchen
        all_skus = daily["sku"].unique().to_list()
        chunk_size = 50
        if len(all_skus) > chunk_size:
            total_chunks = (len(all_skus) + chunk_size - 1) // chunk_size
            print(f"=== Bắt đầu tính Feature Series Streaming cho {len(all_skus):,} SKU ({total_chunks} batches, RAM an toàn < 350 MB) ===")
            total_rows = 0
            rows_by_ind: dict[str, int] = {}
            import gc
            import time

            start_t = time.time()
            for i in range(0, len(all_skus), chunk_size):
                sku_batch = all_skus[i : i + chunk_size]
                sub_daily = daily.filter(pl.col("sku").is_in(sku_batch))
                sub_fs = make_feature_series(daily_agg=sub_daily, window_days=window_days)
                if start is not None:
                    sub_fs = sub_fs.filter(pl.col("window_end") >= start)
                if end is not None:
                    sub_fs = sub_fs.filter(pl.col("window_end") <= end)
                if not sub_fs.is_empty():
                    total_rows += sub_fs.height
                    for ind, count in sub_fs.group_by("indicator_id").len().iter_rows():
                        rows_by_ind[str(ind)] = rows_by_ind.get(str(ind), 0) + count
                    write_table(sub_fs, Table.FEATURE_SERIES, cfg)
                done = min(i + len(sku_batch), len(all_skus))
                pct = (done / len(all_skus)) * 100
                elapsed = time.time() - start_t
                rate = done / elapsed if elapsed > 0 else 0
                eta = (len(all_skus) - done) / rate if rate > 0 else 0
                print(f"  [Features] {done:,}/{len(all_skus):,} SKU ({pct:.1f}% - {total_rows:,} dòng, ~{rate:.1f} SKU/s, ETA: {eta:.0f}s)...")
                del sub_daily, sub_fs
                gc.collect()

            print(f"-> Hoàn tất Feature Series cho toàn bộ {len(all_skus):,} SKU trong {time.time() - start_t:.1f}s!")
            return pl.DataFrame({
                "feature_series": [total_rows] * len(rows_by_ind),
                "skus": [len(all_skus)] * len(rows_by_ind),
                "indicator_id": list(rows_by_ind.keys()),
                "len": list(rows_by_ind.values()),
            })
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


def _get_first_dates_for_category(cfg: Config, category: str) -> pl.DataFrame | None:
    """Quét ngày đầu tiên xuất hiện review của mỗi SKU từ clean_reviews (0.2s) để gọt bỏ số 0 vô nghĩa."""
    try:
        clean_dir = cfg.path("data_interim") / "clean_reviews" / f"category={category}"
        if not clean_dir.exists():
            return None
        files = list(clean_dir.glob("*.parquet"))
        if not files:
            return None
        return (
            pl.scan_parquet(str(clean_dir / "*.parquet"))
            .group_by("sku")
            .agg(pl.col("date").min().alias("first_date"))
            .collect()
        )
    except Exception as e:
        log.warning("Không thể lấy first_dates từ clean_reviews cho '%s': %s", category, e)
        return None


def _detect_batch_worker(
    sku_batch: list[str],
    category: str,
    cfg: Config,
    first_dates: pl.DataFrame | None,
    start: date | None,
    end: date | None,
) -> pl.DataFrame:
    """Worker tính toán chỉ báo I1-I11 độc lập trên 1 nhân CPU."""
    import warnings  # noqa: PLC0415

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    from brandsentinel.detection.stage import detect  # noqa: PLC0415

    sub_fs = read_table(Table.FEATURE_SERIES, cfg, category=category, skus=sku_batch)
    if sub_fs.is_empty():
        return pl.DataFrame()

    # Tối ưu 1: Gọt bỏ chuỗi số 0 trước ngày xuất hiện đầu tiên của từng SKU
    if first_dates is not None and not first_dates.is_empty():
        sub_fs = (
            sub_fs.join(first_dates, on="sku", how="left")
            .filter(pl.col("first_date").is_null() | (pl.col("window_end") >= pl.col("first_date")))
            .drop("first_date")
        )

    if sub_fs.is_empty():
        return pl.DataFrame()

    sub_matrix = detect(sub_fs, cfg)
    if start is not None:
        sub_matrix = sub_matrix.filter(pl.col("window_end") >= start)
    if end is not None:
        sub_matrix = sub_matrix.filter(pl.col("window_end") <= end)
    return sub_matrix


def _score_batch_worker(
    sku_batch: list[str],
    category: str,
    cfg: Config,
    start: date | None,
    end: date | None,
) -> pl.DataFrame:
    """Worker tính toán chấm điểm và phân mức cảnh báo trên 1 nhân CPU."""
    from brandsentinel.scoring.score import score_indicators  # noqa: PLC0415

    sub_m = read_table(Table.INDICATOR_MATRIX, cfg, category=category, skus=sku_batch)
    if sub_m.is_empty():
        return pl.DataFrame()

    sub_alerts = score_indicators(sub_m, cfg)
    if start is not None:
        sub_alerts = sub_alerts.filter(pl.col("window_end") >= start)
    if end is not None:
        sub_alerts = sub_alerts.filter(pl.col("window_end") <= end)
    return sub_alerts


def run_detect(
    category: str,
    *,
    start: date | None = None,
    end: date | None = None,
    config: Config | None = None,
    max_workers: int | None = None,
    resume: bool = True,
) -> pl.DataFrame:
    """Read the full feature history, detect I1-I11, and persist the requested date range."""
    cfg = config or load_config()
    if start and end and start > end:
        raise ValueError("start phải trước hoặc bằng end")

    from brandsentinel.detection.stage import detect  # noqa: PLC0415

    base = cfg.path("data_processed") / "feature_series" / f"category={category}"
    files = list(base.glob("*.parquet")) if base.exists() else []
    if not files:
        raise FileNotFoundError(
            f"Chua co feature_series cua '{category}': "
            f"chay `bs run --stage features --category {category}` truoc"
        )

    # 1. Quét trước ngày xuất hiện đầu tiên của từng SKU để gọt bỏ số 0 vô nghĩa
    first_dates = _get_first_dates_for_category(cfg, category)
    if first_dates is not None:
        log.info("Đã nạp first_dates cho %d SKU để gọt bỏ số 0 dẫn đầu", first_dates.height)

    # Check number of SKUs via lazy scan
    lf = pl.scan_parquet(str(base / "*.parquet"))
    all_skus = sorted(lf.select("sku").unique().collect()["sku"].to_list())
    chunk_size = 15
    workers = max_workers or min(os.cpu_count() or 4, 8)

    # 2. Quét các SKU đã tính toán xong trong indicator_matrix để hỗ trợ Resume
    completed_skus: set[str] = set()
    ind_base = cfg.path("data_processed") / "indicator_matrix" / f"category={category}"
    if resume and ind_base.exists() and list(ind_base.glob("*.parquet")):
        try:
            completed_skus = set(
                pl.scan_parquet(str(ind_base / "*.parquet"))
                .select("sku")
                .unique()
                .collect()["sku"]
                .to_list()
            )
        except Exception as e:
            log.warning("Không thể đọc indicator_matrix hiện có để resume: %s", e)

    target_skus = [s for s in all_skus if s not in completed_skus] if resume else all_skus

    if resume and completed_skus:
        if not target_skus:
            print(
                f"=== [Resume] Toàn bộ {len(all_skus):,} SKU của '{category}' "
                f"đã hoàn tất trong indicator_matrix. Bỏ qua giai đoạn detect ==="
            )
            matrix = read_table(Table.INDICATOR_MATRIX, cfg, category=category)
            if start is not None:
                matrix = matrix.filter(pl.col("window_end") >= start)
            if end is not None:
                matrix = matrix.filter(pl.col("window_end") <= end)
            return matrix

        print(
            f"=== [Resume] Phát hiện {len(completed_skus):,}/{len(all_skus):,} SKU đã hoàn thành. "
            f"Tiếp tục xử lý {len(target_skus):,} SKU còn lại ==="
        )

    if len(all_skus) > 50:
        chunks = [target_skus[i : i + chunk_size] for i in range(0, len(target_skus), chunk_size)]
        total_chunks = len(chunks)
        print(
            f"=== Bắt đầu Detect Parallel cho {len(target_skus):,} SKU "
            f"({total_chunks} batches, {workers} nhân CPU, Zero-Trimmed) ==="
        )
        total_rows = 0
        rows_by_ind: dict[str, int] = {}
        buffer: list[pl.DataFrame] = []
        buffer_rows = 0
        done_skus = len(completed_skus) if resume else 0
        start_t = time.time()

        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _detect_batch_worker,
                    sku_batch,
                    category,
                    cfg,
                    first_dates,
                    start,
                    end,
                ): sku_batch
                for sku_batch in chunks
            }
            for future in as_completed(futures):
                sku_batch = futures[future]
                sub_matrix = future.result()
                done_skus += len(sku_batch)
                if not sub_matrix.is_empty():
                    total_rows += sub_matrix.height
                    buffer.append(sub_matrix)
                    buffer_rows += sub_matrix.height
                    for ind, count in sub_matrix.group_by("indicator_id").len().iter_rows():
                        rows_by_ind[str(ind)] = rows_by_ind.get(str(ind), 0) + count

                # Ghi đệm định kỳ mỗi ~100k dòng để tối ưu I/O ghi đĩa
                if buffer_rows >= 100_000:
                    combined = pl.concat(buffer)
                    write_table(combined, Table.INDICATOR_MATRIX, cfg)
                    buffer.clear()
                    buffer_rows = 0

                pct = (done_skus / len(all_skus)) * 100
                elapsed = time.time() - start_t
                newly_done = done_skus - (len(completed_skus) if resume else 0)
                rate = newly_done / elapsed if elapsed > 0 else 0
                eta = (len(all_skus) - done_skus) / rate if rate > 0 else 0
                print(
                    f"  [Detect Parallel] {done_skus:,}/{len(all_skus):,} SKU "
                    f"({pct:.1f}% - +{total_rows:,} dòng, ~{rate:.2f} SKU/s, ETA: {eta:.0f}s)..."
                )

        # Ghi hết phần buffer còn lại
        if buffer:
            combined = pl.concat(buffer)
            write_table(combined, Table.INDICATOR_MATRIX, cfg)
            buffer.clear()

        print(f"-> Hoàn tất Detect cho toàn bộ {len(all_skus):,} SKU trong {time.time() - start_t:.1f}s!")
        return pl.DataFrame({
            "indicator_matrix": [total_rows] * max(len(rows_by_ind), 1),
            "skus": [len(all_skus)] * max(len(rows_by_ind), 1),
            "indicator_id": list(rows_by_ind.keys()) if rows_by_ind else ["total"],
            "len": list(rows_by_ind.values()) if rows_by_ind else [0],
        })
    else:
        feature_series = read_table(Table.FEATURE_SERIES, cfg, category=category, skus=target_skus)
        if first_dates is not None and not first_dates.is_empty():
            feature_series = (
                feature_series.join(first_dates, on="sku", how="left")
                .filter(pl.col("first_date").is_null() | (pl.col("window_end") >= pl.col("first_date")))
                .drop("first_date")
            )
        matrix = detect(feature_series, cfg)
        if start is not None:
            matrix = matrix.filter(pl.col("window_end") >= start)
        if end is not None:
            matrix = matrix.filter(pl.col("window_end") <= end)
        if matrix.is_empty():
            log.warning("detect: khong co dong nao trong [%s, %s] cho '%s'", start, end, category)
            return matrix
        write_table(matrix, Table.INDICATOR_MATRIX, cfg)
        return matrix


def run_score(
    category: str,
    *,
    start: date | None = None,
    end: date | None = None,
    config: Config | None = None,
    max_workers: int | None = None,
) -> pl.DataFrame:
    """Read indicator history, score it with streaming parallelization, and persist the alert range."""
    cfg = config or load_config()
    if start and end and start > end:
        raise ValueError("start phải trước hoặc bằng end")

    from brandsentinel.core.schemas import empty_table  # noqa: PLC0415
    from brandsentinel.scoring.score import score_indicators  # noqa: PLC0415

    base = cfg.path("data_processed") / "indicator_matrix" / f"category={category}"
    files = list(base.glob("*.parquet")) if base.exists() else []
    if not files:
        raise FileNotFoundError(
            f"Chua co indicator_matrix cua '{category}': "
            f"chay `bs run --stage detect --category {category}` truoc"
        )

    # Check number of SKUs via lazy scan
    lf = pl.scan_parquet(str(base / "*.parquet"))
    all_skus = sorted(lf.select("sku").unique().collect()["sku"].to_list())
    chunk_size = 50
    workers = max_workers or min(os.cpu_count() or 4, 8)

    if len(all_skus) > 50:
        chunks = [all_skus[i : i + chunk_size] for i in range(0, len(all_skus), chunk_size)]
        total_chunks = len(chunks)
        print(
            f"=== Bắt đầu Score Parallel cho {len(all_skus):,} SKU "
            f"({total_chunks} batches, {workers} nhân CPU, RAM an toàn < 350 MB) ==="
        )
        total_rows = 0
        rows_by_level: dict[str, int] = {}
        buffer: list[pl.DataFrame] = []
        buffer_rows = 0
        done_skus = 0
        start_t = time.time()

        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _score_batch_worker,
                    sku_batch,
                    category,
                    cfg,
                    start,
                    end,
                ): sku_batch
                for sku_batch in chunks
            }
            for future in as_completed(futures):
                sku_batch = futures[future]
                sub_alerts = future.result()
                done_skus += len(sku_batch)
                if not sub_alerts.is_empty():
                    total_rows += sub_alerts.height
                    buffer.append(sub_alerts)
                    buffer_rows += sub_alerts.height
                    for lvl, count in sub_alerts.group_by("level").len().iter_rows():
                        rows_by_level[str(lvl)] = rows_by_level.get(str(lvl), 0) + count

                if buffer_rows >= 50_000:
                    combined = pl.concat(buffer)
                    write_table(combined, Table.ALERTS, cfg)
                    buffer.clear()
                    buffer_rows = 0

                pct = (done_skus / len(all_skus)) * 100
                elapsed = time.time() - start_t
                rate = done_skus / elapsed if elapsed > 0 else 0
                eta = (len(all_skus) - done_skus) / rate if rate > 0 else 0
                print(
                    f"  [Score Parallel] {done_skus:,}/{len(all_skus):,} SKU "
                    f"({pct:.1f}% - {total_rows:,} alerts, ~{rate:.2f} SKU/s, ETA: {eta:.0f}s)..."
                )

        if buffer:
            combined = pl.concat(buffer)
            write_table(combined, Table.ALERTS, cfg)
            buffer.clear()

        print(f"-> Hoàn tất Score cho toàn bộ {len(all_skus):,} SKU trong {time.time() - start_t:.1f}s!")
        if total_rows == 0:
            log.warning("score: khong co alert nao trong [%s, %s] cho '%s'", start, end, category)
            return empty_table(Table.ALERTS)
        return read_table(Table.ALERTS, cfg, category=category)
    else:
        matrix = read_table(Table.INDICATOR_MATRIX, cfg, category=category)
        alerts = score_indicators(matrix, cfg)
        if start is not None:
            alerts = alerts.filter(pl.col("window_end") >= start)
        if end is not None:
            alerts = alerts.filter(pl.col("window_end") <= end)
        if alerts.is_empty():
            log.warning("score: khong co alert nao trong [%s, %s] cho '%s'", start, end, category)
            return alerts
        write_table(alerts, Table.ALERTS, cfg)
        return alerts


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


def run_explain(
    category: str,
    *,
    start: date | None = None,
    end: date | None = None,
    config: Config | None = None,
    force_fallback: bool = False,
) -> list[dict[str, Any]]:
    """Đọc alerts mức HIGH và clean_reviews của category, sinh khuyến nghị SCCT (Stage explain)."""
    cfg = config or load_config()
    try:
        alerts = read_table(Table.ALERTS, cfg, category=category)
        clean = read_table(Table.CLEAN_REVIEWS, cfg, category=category)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Chưa có alerts/clean_reviews của '{category}': "
            f"hãy chạy stage `score` và `preprocess` trước"
        ) from e

    nlp = _read_or_none(Table.NLP_FEATURES, cfg, category)

    from brandsentinel.core.io import recommendation_path, write_json
    from brandsentinel.llm.client import call_llm_with_guardrails
    from brandsentinel.llm.evidence import extract_high_alerts_evidence_packs

    if start is not None:
        alerts = alerts.filter(pl.col("window_end") >= start)
    if end is not None:
        alerts = alerts.filter(pl.col("window_end") <= end)

    packs = extract_high_alerts_evidence_packs(
        alerts, clean, nlp_features_df=nlp, cfg=cfg, min_level="HIGH"
    )

    recs: list[dict[str, Any]] = []
    for pack in packs:
        rec = call_llm_with_guardrails(pack, cfg, force_fallback=force_fallback)
        out_dict = rec.model_dump()
        recs.append(out_dict)
        w_end = date.fromisoformat(pack.window_end)
        path = recommendation_path(cfg, category, pack.sku, w_end)
        write_json(path, out_dict)
        if not force_fallback and not rec.is_fallback:
            time.sleep(3.5)

    summary_path = (
        cfg.path("data_processed")
        / "recommendations"
        / f"category={category}"
        / "recommendations.json"
    )
    write_json(summary_path, recs)
    return recs


def run_stage(
    stage: str,
    category: str,
    *,
    start: date | None = None,
    end: date | None = None,
    source: str | Path | None = None,
    with_nlp: bool = True,
    config: Config | None = None,
    resume: bool = True,
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
        fs = run_features(
            category,
            start=start,
            end=end,
            config=cfg,
            with_nlp=with_nlp,
        )
        if "feature_series" in fs.columns and "len" in fs.columns:
            return {
                "feature_series": fs["feature_series"][0],
                "skus": fs["skus"][0],
                "rows_by_indicator": dict(zip(fs["indicator_id"], fs["len"], strict=True)),
            }
        by = fs.group_by("indicator_id").len().sort("indicator_id")
        return {
            "feature_series": fs.height,
            "skus": fs["sku"].n_unique() if fs.height else 0,
            "rows_by_indicator": dict(zip(by["indicator_id"], by["len"], strict=True)),
        }
    if stage == "detect":
        matrix = run_detect(category, start=start, end=end, config=cfg, resume=resume)
        if "indicator_matrix" in matrix.columns and "len" in matrix.columns:
            return {
                "indicator_matrix": matrix["indicator_matrix"][0],
                "skus": matrix["skus"][0],
                "rows_by_indicator": dict(zip(matrix["indicator_id"], matrix["len"], strict=True)),
            }
        by = matrix.group_by("indicator_id").len().sort("indicator_id") if matrix.height else None
        return {
            "indicator_matrix": matrix.height,
            "skus": matrix["sku"].n_unique() if matrix.height else 0,
            "rows_by_indicator": (
                dict(zip(by["indicator_id"], by["len"], strict=True)) if by is not None else {}
            ),
        }
    if stage == "score":
        alerts = run_score(category, start=start, end=end, config=cfg)
        by_level = alerts.group_by("level").len().sort("level") if alerts.height else None
        return {
            "alerts": alerts.height,
            "skus": alerts["sku"].n_unique() if alerts.height else 0,
            "rows_by_level": (
                dict(zip(by_level["level"], by_level["len"], strict=True))
                if by_level is not None
                else {}
            ),
        }
    if stage == "explain":
        recs = run_explain(category, start=start, end=end, config=cfg)
        return {
            "recommendations": len(recs),
            "skus": len({r["sku"] for r in recs}),
            "fallback_count": sum(1 for r in recs if r.get("is_fallback")),
        }
    raise NotImplementedError(stage)
