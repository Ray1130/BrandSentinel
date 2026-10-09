"""Dữ liệu mock cho cả 6 bảng, ĐÚNG hợp đồng dữ liệu, để P1/P2/P3 làm việc song song.

    from brandsentinel.testing.mock_data import make_all
    tables = make_all(seed=42)            # dict[Table, pl.DataFrame]
    tables[Table.CLEAN_REVIEWS]

Kịch bản có sẵn (SKU đánh số B0MOCK0000, B0MOCK0001, ...):
  - SKU 0: khủng hoảng ở `crisis_days` ngày cuối (volume tăng, rating xấu, nội dung an toàn).
  - SKU 1: review bombing ở `bombing_days` ngày cuối (volume tăng, tỷ lệ verified thấp, review mẫu).
  - Các SKU còn lại: bình thường, có chu kỳ tuần (cuối tuần nhiều review hơn).

LƯU Ý: đặc trưng, chỉ báo và cảnh báo trong file này được tính theo cách ĐƠN GIẢN chỉ để có dữ liệu
mock hợp lệ; đừng dùng làm cài đặt thật (cài đặt thật nằm ở features/, detection/, scoring/).
"""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import UTC, date, datetime, timedelta

import numpy as np
import polars as pl

from brandsentinel.core.config import Config, get_config
from brandsentinel.core.ids import review_id
from brandsentinel.core.schemas import conform, validate
from brandsentinel.core.types import (
    ASPECTS,
    FEATURES_BY_INDICATOR,
    INDICATOR_IDS,
    SCORED_GROUPS,
    Level,
    Table,
)

WEEK_FACTOR = [1.0, 1.0, 1.0, 1.0, 1.15, 1.3, 1.2]  # thứ Hai .. Chủ nhật
RATING_P_NORMAL = [0.05, 0.05, 0.10, 0.25, 0.55]
RATING_P_CRISIS = [0.35, 0.20, 0.10, 0.10, 0.25]
RISK_TERMS = (
    "overheat", "burning smell", "caught fire", "rash", "allergic",
    "injury", "recall", "lawsuit", "fake", "toxic",
)  # fmt: skip
ASPECT_KEYWORDS = {
    "safety": ("overheat", "burning", "fire", "rash", "allergic", "injury", "toxic"),
    "delivery": ("late", "shipping", "delivery", "arrived"),
    "refund": ("refund", "return"),
    "quality": ("quality", "stopped working", "battery", "scam"),
}
TEXT_POS = [
    "works great, love it", "good quality for the price", "fast delivery, as described",
    "perfect, my family loves it", "excellent value, would buy again",
]  # fmt: skip
TEXT_MID = ["okay product, nothing special", "average, does the job"]
TEXT_NEG = [
    "stopped working after a month", "poor quality, want a refund", "arrived late and damaged",
]  # fmt: skip
TEXT_CRISIS = [
    "it started to overheat and there was a burning smell",
    "battery overheat, almost caught fire, dangerous",
    "my baby got a rash after using it",
]
TEXT_BOMB = {1: "terrible scam do not buy", 5: "best product ever buy now"}


# ---------------------------------------------------------------- clean_reviews
def make_clean_reviews(
    *,
    n_skus: int = 6,
    days: int = 120,
    start: date = date(2023, 1, 2),
    seed: int = 42,
    category: str = "Mock_Category",
    crisis_sku_index: int | None = 0,
    rating_only_sku_index: int | None = None,
    bombing_sku_index: int | None = 1,
    crisis_days: int = 14,
    rating_only_days: int = 14,
    bombing_days: int = 10,
) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    cols: dict[str, list] = {
        k: []
        for k in (
            "review_id", "sku", "category", "user_id", "ts", "date", "rating", "text_raw",
            "text_norm", "verified_purchase", "is_spam", "spam_reason",
        )
    }  # fmt: skip

    def add(sku, ts, rating, text, verified, spam, reason):
        user = f"U{int(rng.integers(0, 10**7)):07d}"
        ts_ms = int(ts.timestamp() * 1000)
        cols["review_id"].append(review_id(sku, user, ts_ms))
        cols["sku"].append(sku)
        cols["category"].append(category)
        cols["user_id"].append(user)
        cols["ts"].append(ts)
        cols["date"].append(ts.date())
        cols["rating"].append(int(rating))
        cols["text_raw"].append(text)
        cols["text_norm"].append(text.lower())
        cols["verified_purchase"].append(bool(verified))
        cols["is_spam"].append(bool(spam))
        cols["spam_reason"].append(reason)

    def pick_text(rating, crisis):
        if rating >= 4:
            return str(rng.choice(TEXT_POS))
        if rating == 3:
            return str(rng.choice(TEXT_MID))
        pool = TEXT_CRISIS if crisis and rng.random() < 0.7 else TEXT_NEG
        return str(rng.choice(pool))

    for s in range(n_skus):
        sku = f"B0MOCK{s:04d}"
        base = float(rng.uniform(8, 20))
        for d in range(days):
            day = start + timedelta(days=d)
            lam = base * WEEK_FACTOR[day.weekday()]
            crisis = s == crisis_sku_index and d >= days - crisis_days
            rating_only = s == rating_only_sku_index and d >= days - rating_only_days
            bomb = s == bombing_sku_index and d >= days - bombing_days
            if crisis:
                lam *= 2.0
            midnight = datetime(day.year, day.month, day.day, tzinfo=UTC)
            for _ in range(int(rng.poisson(lam))):
                rating = int(
                    rng.choice(
                        [1, 2, 3, 4, 5],
                        p=RATING_P_CRISIS if crisis or rating_only else RATING_P_NORMAL,
                    )
                )
                ts = midnight + timedelta(
                    seconds=int(rng.integers(0, 86400)), milliseconds=int(rng.integers(0, 1000))
                )
                text = pick_text(rating, crisis)
                if rng.random() < 0.01:  # spam nền: review quá ngắn
                    add(sku, ts, rating, "ok", rng.random() < 0.9, True, "short_text")
                else:
                    add(sku, ts, rating, text, rng.random() < 0.9, False, None)
            if bomb:
                for _ in range(int(rng.poisson(lam * 3))):
                    rating = 1 if rng.random() < 0.6 else 5
                    ts = midnight + timedelta(
                        seconds=int(rng.integers(0, 86400)), milliseconds=int(rng.integers(0, 1000))
                    )
                    spam = rng.random() < 0.7
                    add(
                        sku,
                        ts,
                        rating,
                        TEXT_BOMB[rating],
                        rng.random() < 0.15,
                        spam,
                        "template_repeat" if spam else None,
                    )

    df = pl.DataFrame(cols, schema_overrides={"ts": pl.Datetime("us", "UTC")})
    return conform(
        Table.CLEAN_REVIEWS, df.unique(subset="review_id", keep="first", maintain_order=True)
    )


# ---------------------------------------------------------------- daily_agg
def make_daily_agg(clean: pl.DataFrame) -> pl.DataFrame:
    """Gom SKU x ngày, điền 0 cho ngày thiếu (bản đơn giản; bản thật: preprocessing/aggregate)."""
    keys = ["sku", "category", "date"]
    ok = (
        clean.filter(~pl.col("is_spam"))
        .group_by(keys)
        .agg(
            pl.len().alias("n"),
            (pl.col("rating") <= 2).sum().alias("n_neg"),
            *[(pl.col("rating") == k).sum().alias(f"n{k}") for k in range(1, 6)],
            pl.col("rating").cast(pl.Float64).sum().alias("sum_rating"),
            (pl.col("rating").cast(pl.Float64) ** 2).sum().alias("sumsq_rating"),
            pl.col("verified_purchase").fill_null(False).sum().alias("n_verified"),
        )
    )
    allr = clean.group_by(keys).agg(
        pl.len().alias("n_all"),
        pl.col("verified_purchase").fill_null(False).sum().alias("n_verified_all"),
    )
    d0, d1 = clean["date"].min(), clean["date"].max()
    grid = (
        clean.select("sku", "category").unique()
        .join(pl.DataFrame({"date": pl.date_range(d0, d1, "1d", eager=True)}), how="cross")
    )  # fmt: skip
    out = grid.join(ok, on=keys, how="left").join(allr, on=keys, how="left").fill_null(0)
    return conform(Table.DAILY_AGG, out.sort(["sku", "date"]))


# ---------------------------------------------------------------- nlp_features
def make_nlp_features(clean: pl.DataFrame, *, seed: int = 42) -> pl.DataFrame:
    """Sentiment/risk/aspect đơn giản theo từ khóa (bản thật: features/nlp/*)."""
    rng = np.random.default_rng(seed + 1)
    rows: dict[str, list] = {k: [] for k in (
        "review_id", "category", "date", "sentiment_score", "risk_hit", "risk_terms",
        "risk_sim", "aspect", "aspect_sentiment", "mismatch",
    )}  # fmt: skip
    for rid, cat, day, rating, text in clean.select(
        "review_id", "category", "date", "rating", "text_norm"
    ).iter_rows():
        sent = (rating - 3) / 2 + float(rng.normal(0, 0.15))
        if rating == 5 and rng.random() < 0.02:  # review mỉa mai: 5 sao nhưng văn bản tiêu cực
            sent = -0.7
        sent = max(-1.0, min(1.0, sent))
        hits = [t for t in RISK_TERMS if t in text]
        aspect = next(
            (a for a, kws in ASPECT_KEYWORDS.items() if any(k in text for k in kws)), None
        )
        rows["review_id"].append(rid)
        rows["category"].append(cat)
        rows["date"].append(day)
        rows["sentiment_score"].append(sent)
        rows["risk_hit"].append(bool(hits))
        rows["risk_terms"].append(hits or None)
        rows["risk_sim"].append(None)
        rows["aspect"].append(aspect)
        rows["aspect_sentiment"].append(sent if aspect else None)
        rows["mismatch"].append((rating >= 4 and sent < -0.5) or (rating <= 2 and sent > 0.5))
    df = pl.DataFrame(
        rows,
        schema_overrides={
            "risk_terms": pl.List(pl.String),
            "risk_sim": pl.Float32,
            "aspect": pl.String,
        },
    )
    return conform(Table.NLP_FEATURES, df)


# ---------------------------------------------------------------- feature_series
def _wsum(x: np.ndarray, w: int) -> np.ndarray:
    cs = np.concatenate([[0.0], np.cumsum(x, dtype=float)])
    out = np.full(len(x), np.nan)
    out[w - 1 :] = cs[w:] - cs[:-w]
    return out


def make_feature_series(
    clean: pl.DataFrame, daily: pl.DataFrame, nlp: pl.DataFrame, *, window_days: int = 7
) -> pl.DataFrame:
    """Đặc trưng thô theo cửa sổ trượt (bản đơn giản; bản thật: features/*)."""
    w = window_days
    joined = clean.filter(~pl.col("is_spam")).join(
        nlp.select("review_id", "risk_hit", "aspect", "sentiment_score", "mismatch"), on="review_id"
    )
    per_day: dict = defaultdict(lambda: defaultdict(float))
    users: dict = defaultdict(set)
    for sku, day, user, hit, aspect, sent, mism in joined.select(
        "sku", "date", "user_id", "risk_hit", "aspect", "sentiment_score", "mismatch"
    ).iter_rows():
        acc = per_day[(sku, day)]
        acc["nlp_n"] += 1
        acc["hits"] += hit
        acc["mism"] += bool(mism)
        if hit and user is not None:
            users[(sku, day)].add(user)
        if aspect:
            acc[f"asp_n_{aspect}"] += 1
            acc[f"asp_neg_{aspect}"] += sent < 0
    rows = {
        k: []
        for k in (
            "sku",
            "category",
            "window_end",
            "indicator_id",
            "feature",
            "raw_value",
            "n_window",
        )
    }
    feat_to_ind = {f: i for i, fs in FEATURES_BY_INDICATOR.items() for f in fs}

    for (sku,), g in daily.sort("date").partition_by(["sku"], as_dict=True).items():
        cat, dates = g["category"][0], g["date"].to_list()
        a = {
            c: g[c].to_numpy().astype(float)
            for c in g.columns
            if c not in ("sku", "category", "date")
        }

        def get(key, sku=sku, dates=dates):
            return np.array([per_day[(sku, d)][key] for d in dates])

        w_n, w_neg = _wsum(a["n"], w), _wsum(a["n_neg"], w)
        w_n1, w_n3, w_n5 = _wsum(a["n1"], w), _wsum(a["n3"], w), _wsum(a["n5"], w)
        w_sum, w_sq = _wsum(a["sum_rating"], w), _wsum(a["sumsq_rating"], w)
        w_hits, w_mism, w_nlp = _wsum(get("hits"), w), _wsum(get("mism"), w), _wsum(get("nlp_n"), w)
        w_all, w_vall = _wsum(a["n_all"], w), _wsum(a["n_verified_all"], w)
        asp = {x: (_wsum(get(f"asp_neg_{x}"), w), _wsum(get(f"asp_n_{x}"), w)) for x in ASPECTS}

        def ratio(num, den, i):
            return float(num[i] / den[i]) if den[i] > 0 else None

        for i in range(w - 1, len(dates)):
            prior_window_count = w_n[i - w] if i >= 2 * w - 1 else None
            vals: dict[str, float | None] = {
                "log1p_daily_count": math.log1p(a["n"][i]),
                "growth_rate": (
                    float((w_n[i] - prior_window_count) / (prior_window_count + 1.0))
                    if prior_window_count is not None
                    else None
                ),
                "neg_ratio_shrunk": float((w_neg[i] + 0.5 * w_n3[i] + 2) / (w_n[i] + 20)),
                "low_star_ratio_shrunk": float((w_neg[i] + 2) / (w_n[i] + 20)),
                "rating_rolling_variance": (
                    float(w_sq[i] / w_n[i] - (w_sum[i] / w_n[i]) ** 2) if w_n[i] >= 2 else None
                ),
                "extreme_share": ratio(w_n1 + w_n5, w_n, i),
                "share_1star": ratio(w_n1, w_n, i),
                "share_5star": ratio(w_n5, w_n, i),
                "term_burst_score": float(w_hits[i] / 2.0),
                "risk_hits": float(w_hits[i]),
                "risk_distinct_users": float(
                    len(set().union(*(users[(sku, d)] for d in dates[i - w + 1 : i + 1])))
                ),
                "mismatch_rate": ratio(w_mism, w_nlp, i),
                "verified_ratio_all": ratio(w_vall, w_all, i),
                **{f"aspect_neg_{x}": ratio(*asp[x], i) for x in ASPECTS},
            }
            for feat, val in vals.items():
                rows["sku"].append(sku)
                rows["category"].append(cat)
                rows["window_end"].append(dates[i])
                rows["indicator_id"].append(feat_to_ind[feat])
                rows["feature"].append(feat)
                rows["raw_value"].append(val)
                if feat.startswith("aspect_neg_"):
                    aspect = feat.removeprefix("aspect_neg_")
                    rows["n_window"].append(int(asp[aspect][1][i]))
                else:
                    rows["n_window"].append(int(w_n[i]))
    return conform(
        Table.FEATURE_SERIES, pl.DataFrame(rows, schema_overrides={"raw_value": pl.Float64})
    )


# ---------------------------------------------------------------- indicator_matrix
def make_indicator_matrix(
    feature_series: pl.DataFrame, *, baseline_windows: int = 60, k: float = 2.5
) -> pl.DataFrame:
    """Kích hoạt đơn giản bằng robust z so với baseline đầu chuỗi (bản thật: detection/*)."""
    rows = {
        c: [] for c in ("sku", "category", "window_end", "indicator_id", "triggered", "strength")
    }
    for (sku, iid), g in feature_series.partition_by(["sku", "indicator_id"], as_dict=True).items():
        cat = g["category"][0]
        piv = g.pivot(on="feature", index="window_end", values="raw_value").sort("window_end")
        dates = piv["window_end"].to_list()
        primary = FEATURES_BY_INDICATOR[iid][0]
        x = piv[primary].to_numpy().astype(float)
        base = x[:baseline_windows][~np.isnan(x[:baseline_windows])]
        med = float(np.median(base)) if len(base) else 0.0
        mad = float(np.median(np.abs(base - med))) if len(base) else 0.0
        scale = max(1.4826 * mad, 0.05 * abs(med), 1e-3)
        z = (x - med) / scale
        direction = -1.0 if iid == "I11" else 1.0
        users = piv["risk_distinct_users"].to_numpy().astype(float) if iid == "I8" else None
        for j, d in enumerate(dates):
            if np.isnan(x[j]):
                trig, strength = False, None
            elif iid == "I8":
                trig, strength = bool(x[j] >= 2 and users[j] >= 2), float(x[j])
            else:
                zz = float(np.clip(direction * z[j], -50, 50))
                trig, strength = bool(zz > k), zz
            for key, val in zip(rows, (sku, cat, d, iid, trig, strength), strict=True):
                rows[key].append(val)
    return conform(
        Table.INDICATOR_MATRIX, pl.DataFrame(rows, schema_overrides={"strength": pl.Float64})
    )


# ---------------------------------------------------------------- alerts
def make_alerts(indicator_matrix: pl.DataFrame, cfg: Config) -> pl.DataFrame:
    """Chấm điểm và phân mức đơn giản theo thresholds.yaml (bản thật: scoring/*)."""
    ind, th = cfg.indicators, cfg.thresholds
    pts = {i: ind.priority_points[ind.indicators[i].priority] for i in ind.scored_ids}
    total = sum(pts.values())
    group_of = {i: ind.indicators[i].group for i in INDICATOR_IDS}
    tau1, tau2 = th.calibration.fallback.tau1, th.calibration.fallback.tau2
    counted = {g.value for g in SCORED_GROUPS}
    trig: dict = defaultdict(set)
    cats: dict = {}
    for sku, cat, d, iid, t in indicator_matrix.select(
        "sku", "category", "window_end", "indicator_id", "triggered"
    ).iter_rows():
        cats[sku] = cat
        if t:
            trig[(sku, d)].add(iid)
    rows = {
        c: []
        for c in (
            "sku",
            "category",
            "window_end",
            "score",
            "level",
            "groups_active",
            "persist",
            "verify_flag",
            "triggered_ids",
        )
    }
    for sku in sorted(cats):
        persist = 0
        for d in sorted(
            {k[1] for k in trig if k[0] == sku}
            | set(indicator_matrix.filter(pl.col("sku") == sku)["window_end"].unique().to_list())
        ):
            ids = trig[(sku, d)]
            score = sum(pts[i] for i in ids if i in pts) / total
            groups = sorted({group_of[i] for i in ids if i in pts} & counted)
            persist = persist + 1 if score >= tau1 else 0
            severe = "I8" in ids and th.tiering.high.allow_severe_override
            high = score >= tau2 and (len(groups) >= th.tiering.high.min_groups_active or severe)
            medium = score >= tau1 and (
                persist >= th.tiering.medium.min_persist_windows
                or len(groups) >= th.tiering.medium.min_groups_active
            )
            level = Level.HIGH if high else Level.MEDIUM if medium else Level.LOW
            for key, val in zip(
                rows,
                (
                    sku,
                    cats[sku],
                    d,
                    float(score),
                    level.value,
                    groups,
                    min(persist, 100),
                    bool({"I10", "I11"} & ids),
                    sorted(ids, key=lambda s: int(s[1:])),
                ),
                strict=True,
            ):
                rows[key].append(val)
    return conform(
        Table.ALERTS,
        pl.DataFrame(
            rows,
            schema_overrides={
                "groups_active": pl.List(pl.String),
                "triggered_ids": pl.List(pl.String),
            },
        ),
    )


# ---------------------------------------------------------------- tất cả
def make_all(
    *,
    seed: int = 42,
    n_skus: int = 6,
    days: int = 120,
    category: str = "Mock_Category",
    cfg: Config | None = None,
    rating_only_sku_index: int | None = None,
    bombing_sku_index: int | None = 1,
) -> dict[Table, pl.DataFrame]:
    cfg = cfg or get_config()
    clean = make_clean_reviews(
        n_skus=n_skus,
        days=days,
        seed=seed,
        category=category,
        rating_only_sku_index=rating_only_sku_index,
        bombing_sku_index=bombing_sku_index,
    )
    daily = make_daily_agg(clean)
    nlp = make_nlp_features(clean, seed=seed)
    feats = make_feature_series(clean, daily, nlp, window_days=cfg.default.time.window_days)
    matrix = make_indicator_matrix(feats)
    alerts = make_alerts(matrix, cfg)
    tables = {
        Table.CLEAN_REVIEWS: clean,
        Table.DAILY_AGG: daily,
        Table.NLP_FEATURES: nlp,
        Table.FEATURE_SERIES: feats,
        Table.INDICATOR_MATRIX: matrix,
        Table.ALERTS: alerts,
    }
    for t, df in tables.items():
        validate(t, df)
    return tables
