"""Offline, read-only Streamlit dashboard for SKU risk triage."""

from __future__ import annotations

import re
from datetime import date, timedelta
from html import escape
from typing import Any

import altair as alt
import httpx
import polars as pl
import streamlit as st

from brandsentinel.core.config import get_config
from brandsentinel.core.types import Table
from brandsentinel.llm.schemas import RecommendationOutput
from brandsentinel.scoring.thresholds import thresholds_for_category
from brandsentinel.serving import dashboard_data as data

LEVEL_COLORS = {"HIGH": "#c0392b", "MEDIUM": "#e67e22", "LOW": "#28836b"}
GROUP_LABELS = {"volume": "Tần suất", "rating": "Rating", "content": "Nội dung"}
DEPARTMENT_LABELS = {
    "QA_KyThuat": "QA / Kỹ thuật",
    "VanHanh": "Vận hành",
    "CSKH": "CSKH",
    "TruyenThong": "Truyền thông",
}


def _message_block(label: str, fn: Any) -> Any:
    """Keep a failed optional dashboard block visible without stopping other tabs."""
    try:
        return fn()
    except Exception as exc:
        st.error(f"{label}: {exc}")
        return None


def _get_date(value: Any, fallback: date) -> date:
    if isinstance(value, list):
        value = value[0] if value else None
    try:
        return date.fromisoformat(str(value)) if value else fallback
    except ValueError:
        return fallback


def _set_deep_link(category: str, as_of: date, sku: str | None) -> None:
    st.query_params["category"] = category
    st.query_params["as_of"] = as_of.isoformat()
    if sku:
        st.query_params["sku"] = sku
    else:
        st.query_params.pop("sku", None)


def _level_badge(level: str) -> str:
    return f":{ {'HIGH': 'red', 'MEDIUM': 'orange', 'LOW': 'green'}.get(level, 'gray') }[{level}]"


def _alert_history(category: str, cfg: Any) -> pl.DataFrame:
    return data.load_alerts(cfg, category)


def _date_bounds(category: str, cfg: Any, alerts: pl.DataFrame) -> tuple[date, date]:
    if not alerts.is_empty():
        return alerts["window_end"].min(), alerts["window_end"].max()
    daily = data.read_category_table(Table.DAILY_AGG, cfg, category)
    if daily.is_empty():
        today = date.today()
        return today, today
    return daily["date"].min(), daily["date"].max()


def _overview(
    category: str,
    cfg: Any,
    alerts: pl.DataFrame,
    as_of: date,
    limit: int,
    levels: list[str],
    groups: list[str],
    verify_only: bool,
    sku_search: str,
) -> tuple[str | None, pl.DataFrame]:
    stats = data.kpis(alerts, as_of)
    cols = st.columns(4)
    cols[0].metric("SKU HIGH", stats["high"])
    cols[1].metric("SKU MEDIUM", stats["medium"])
    cols[2].metric("Cờ xác minh", stats["verify_flag"])
    cols[3].metric("HIGH mới · 7 ngày", stats["new_high_7d"])
    st.caption(
        "Độ phủ: hợp đồng dữ liệu hiện không lưu trạng thái `insufficient_data` "
        "hay tổng SKU đã chấm; "
        "không thể tính số này mà không suy đoán."
    )

    st.subheader("Số đối tượng theo mức rủi ro")
    if alerts.is_empty():
        st.info("Chưa có dữ liệu cảnh báo trong category này.")
    else:
        trend = (
            alerts.filter(pl.col("window_end") <= as_of)
            .group_by(["window_end", "level"])
            .agg(pl.col("sku").n_unique().alias("sku_count"))
            .to_pandas()
        )
        chart = (
            alt.Chart(trend)
            .mark_area(opacity=0.68)
            .encode(
                x=alt.X("window_end:T", title="Ngày"),
                y=alt.Y("sku_count:Q", stack="zero", title="Số SKU"),
                color=alt.Color(
                    "level:N",
                    title="Mức",
                    scale=alt.Scale(
                        domain=["HIGH", "MEDIUM", "LOW"],
                        range=[LEVEL_COLORS["HIGH"], LEVEL_COLORS["MEDIUM"], LEVEL_COLORS["LOW"]],
                    ),
                ),
                tooltip=["window_end:T", "level:N", "sku_count:Q"],
            )
            .properties(height=240)
        )
        st.altair_chart(chart, use_container_width=True)
        st.caption("LOW không xuất hiện trong biểu đồ vì bảng `alerts` chỉ lưu MEDIUM và HIGH.")

    st.subheader("Danh sách ưu tiên")
    ranked = data.priority_table(alerts, as_of, limit, step_days=cfg.default.time.step_days)
    mask = pl.lit(True)
    if levels:
        mask &= pl.col("level").is_in(levels)
    if groups:
        mask &= pl.col("groups_active").list.eval(pl.element().is_in(groups)).list.any()
    if verify_only:
        mask &= pl.col("verify_flag")
    if sku_search:
        mask &= pl.col("sku").str.contains(sku_search, literal=True, strict=False)
    ranked = ranked.filter(mask) if not ranked.is_empty() else ranked
    if ranked.is_empty():
        st.info("Chưa có SKU ở mức MEDIUM/HIGH tại ngày xét.")
        return None, ranked

    thresholds = thresholds_for_category(category, cfg)
    rec_summaries = (
        _message_block(
            "Không đọc được tóm tắt khuyến nghị",
            lambda: data.recommendation_summaries(cfg, category),
        )
        or {}
    )
    table_rows = []
    for row in ranked.to_dicts():
        groups = row["active_groups"]
        group_marks = " · ".join(
            f"{'●' if group in groups else '○'} {GROUP_LABELS.get(group, group)}"
            for group in ("volume", "rating", "content")
        )
        table_rows.append(
            {
                "SKU": row["sku"],
                "Category": row["category"],
                "Mức": row["level"],
                "Điểm": float(row["score"]),
                "Chỉ báo": f"{row['active_indicator_count']}/9",
                "Nhóm active": group_marks,
                "Cửa sổ": int(row["persist"]),
                "Cờ xác minh": "Có" if row["verify_flag"] else "Không",
                "Ở mức này từ": row["level_since"],
                "Ngưỡng τ₁ / τ₂": f"{thresholds.tau1:.2f} / {thresholds.tau2:.2f}",
                "Khuyến nghị": rec_summaries.get((row["sku"], row["window_end"].isoformat()), ""),
            }
        )
    frame = pl.DataFrame(table_rows).to_pandas()
    selection = st.dataframe(
        frame,
        hide_index=True,
        use_container_width=True,
        on_select="rerun",
        selection_mode="single-row",
        column_config={
            "Mức": st.column_config.TextColumn(
                "Mức", help="Điểm là thứ tự ưu tiên, không phải xác suất."
            ),
            "Điểm": st.column_config.ProgressColumn(
                "Điểm",
                help=(
                    "Điểm để xếp hạng ưu tiên kiểm tra; "
                    f"τ₁={thresholds.tau1:.2f}, τ₂={thresholds.tau2:.2f}."
                ),
                min_value=0.0,
                max_value=1.0,
                format="%.2f",
            ),
        },
    )
    chosen = st.session_state.get("dashboard_selected_sku")
    selected_rows = selection.selection.rows if selection and selection.selection else []
    if selected_rows:
        chosen = str(frame.iloc[selected_rows[0]]["SKU"])
        st.session_state.dashboard_selected_sku = chosen
    valid_skus = set(ranked["sku"].to_list())
    if chosen not in valid_skus:
        chosen = ranked["sku"][0]
        st.session_state.dashboard_selected_sku = chosen
    return chosen, ranked


def _indicator_card(record: dict[str, Any]) -> None:
    marker = "● Đang kích hoạt" if record["triggered"] else "○ Không kích hoạt"
    st.markdown(f"**{record['indicator_id']} · {record['name']}**")
    strength = record["strength"] if record["strength"] is not None else "—"
    st.caption(f"{marker} · Độ mạnh: {strength}")
    if record["group"] != "flag":
        st.caption(f"Trọng số {record['weight']:.3f} · Đóng góp {record['contribution']:.3f}")
        st.progress(min(1.0, record["contribution"] / max(record["weight"], 1e-12)))
    else:
        st.caption(f"Nhóm kiểm tra độ tin cậy · Ưu tiên {record['priority']}")


def _render_detail(
    category: str,
    cfg: Any,
    alerts: pl.DataFrame,
    ranked: pl.DataFrame,
    sku: str,
    as_of: date,
) -> None:
    record_frame = ranked.filter(pl.col("sku") == sku)
    if record_frame.is_empty():
        st.info("Chọn một SKU trong danh sách ưu tiên để mở chi tiết.")
        return
    record = record_frame.to_dicts()[0]
    thresholds = thresholds_for_category(category, cfg)
    st.header(f"Chi tiết SKU · {sku}")
    cols = st.columns(4)
    cols[0].markdown(f"**Mức**  \n{_level_badge(record['level'])}")
    cols[1].metric("Điểm ưu tiên", f"{record['score']:.3f}")
    cols[2].metric("τ₁ / τ₂", f"{thresholds.tau1:.2f} / {thresholds.tau2:.2f}")
    cols[3].metric("Ở mức này từ", record["level_since"].isoformat())
    st.caption(
        f"Snapshot đến {as_of.isoformat()} · cảnh báo gần nhất ghi nhận "
        f"{record['window_end'].isoformat()}."
    )
    st.caption("Điểm 0–1 dùng để xếp hạng ưu tiên kiểm tra, không phải xác suất.")
    if record["verify_flag"]:
        st.warning("Cần xác minh tính xác thực (I10/I11).")
    if record["score"] >= thresholds.tau2 and record["level"] != "HIGH":
        st.warning("Đủ điểm, thiếu xác nhận chéo.")
    elif (
        record["level"] == "HIGH"
        and "I8" in record["triggered_ids"]
        and len(record["groups_active"]) < cfg.thresholds.tiering.high.min_groups_active
    ):
        st.warning("Lên HIGH do bằng chứng an toàn nghiêm trọng (I8).")

    st.subheader("Vì sao hệ thống cảnh báo?")
    breakdown = _message_block(
        "Lưới chỉ báo bị lỗi",
        lambda: _render_indicators(cfg, category, sku, record["window_end"]),
    )
    breakdown = breakdown or []

    st.subheader("Diễn biến theo thời gian")
    range_start = max(date.min, as_of - timedelta(days=89))
    _message_block(
        "Diễn biến theo thời gian bị lỗi",
        lambda: _render_series(
            data.series_for(cfg, category, sku, range_start, as_of), range_start, as_of
        ),
    )
    _message_block("Bằng chứng bị lỗi", lambda: _render_evidence(cfg, category, record, breakdown))
    st.subheader("Khuyến nghị")
    if record["level"] == "HIGH":
        _message_block(
            "Khuyến nghị",
            lambda: _render_recommendation(
                data.load_recommendation(cfg, category, sku, record["window_end"])
            ),
        )
    else:
        st.info("Khuyến nghị SCCT được tạo cho SKU ở mức HIGH.")


def _render_indicators(cfg: Any, category: str, sku: str, as_of: date) -> list[dict[str, Any]]:
    matrix = data.read_category_table(
        Table.INDICATOR_MATRIX, cfg, category, start=as_of, end=as_of, skus=[sku]
    )
    breakdown = data.indicator_breakdown(matrix, cfg, sku, as_of)
    for group in ("volume", "rating", "content"):
        group_rows = [item for item in breakdown if item["group"] == group]
        st.markdown(f"**{GROUP_LABELS[group]}**")
        columns = st.columns(3)
        for index, item in enumerate(group_rows):
            with columns[index % 3]:
                _indicator_card(item)
    st.markdown("**Kiểm tra độ tin cậy · không cộng vào điểm**")
    flag_cols = st.columns(2)
    for index, item in enumerate(item for item in breakdown if item["group"] == "flag"):
        with flag_cols[index % 2]:
            _indicator_card(item)
    return breakdown


def _render_evidence(
    cfg: Any, category: str, record: dict[str, Any], breakdown: list[dict[str, Any]]
) -> None:
    st.subheader("Bằng chứng")
    triggered = ["Tất cả"] + [item["indicator_id"] for item in breakdown if item["triggered"]]
    filter_value = st.selectbox("Lọc review theo chỉ báo", triggered, key="evidence_indicator")
    evidence_indicator = None if filter_value == "Tất cả" else filter_value
    all_reviews = data.evidence_reviews(cfg, category, record)
    reviews = all_reviews
    if evidence_indicator == "I8":
        reviews = [review for review in reviews if review["risk_hit"]]
    elif evidence_indicator == "I3":
        reviews = [
            review
            for review in reviews
            if review["sentiment_score"] is not None and review["sentiment_score"] < 0
        ]
    elif evidence_indicator == "I4":
        reviews = [review for review in reviews if review["rating"] <= 2]
    elif evidence_indicator == "I6":
        reviews = [review for review in reviews if review["rating"] in (1, 5)]
    elif evidence_indicator == "I7":
        reviews = [review for review in reviews if review["risk_hit"]]
    elif evidence_indicator == "I9":
        reviews = [review for review in reviews if review["aspect"]]
    elif evidence_indicator == "I10":
        reviews = [review for review in reviews if review["mismatch"]]
    elif evidence_indicator == "I11":
        reviews = [review for review in reviews if review["verified_purchase"] is False]
    for review in all_reviews:
        st.markdown(
            f'<span id="review-{escape(str(review["review_id"]), quote=True)}"></span>',
            unsafe_allow_html=True,
        )
    if reviews:
        evidence_frame = pl.DataFrame(
            [
                {
                    "Ngày": review["date"],
                    "Sao": review["rating"],
                    "Verified": review["verified_purchase"],
                    "Sentiment": review["sentiment_score"],
                    "Cờ spam": review["is_spam"],
                    "Review ID": review["review_id"],
                    "Từ khóa rủi ro": ", ".join(review["risk_terms"]),
                    "Khía cạnh": review["aspect"],
                    "Văn bản (đã che PII)": review["text"],
                }
                for review in reviews
            ]
        ).to_pandas()
        st.dataframe(evidence_frame, hide_index=True, use_container_width=True)
        for review in reviews:
            review_id = str(review["review_id"])
            text = escape(review["text"])
            for term in sorted(review["risk_terms"], key=len, reverse=True):
                if term:
                    text = re.sub(
                        re.escape(escape(term)),
                        lambda match: f"<mark>{match.group(0)}</mark>",
                        text,
                        flags=re.IGNORECASE,
                    )
            st.markdown(f"**{review_id}**  \n{text}", unsafe_allow_html=True)
    else:
        st.info("Không có review đại diện cho chỉ báo/cửa sổ này.")


def _render_series(
    series: dict[str, pl.DataFrame],
    start: date,
    as_of: date,
) -> None:
    daily = series["daily"]
    features = series["features"]
    alerts = series["alerts"]
    indicators = series["indicators"]
    if daily.is_empty() and features.is_empty():
        st.info("Chưa có chuỗi chi tiết cho SKU này.")
        return
    if not daily.is_empty():
        daily_chart = _context_chart(
            daily.select("date", "n").to_pandas(),
            "n",
            "Số review theo ngày",
            {"I1", "I2"},
            alerts,
            indicators,
            start,
            as_of,
        )
        st.altair_chart(daily_chart, use_container_width=True)
    display_features = [
        ("neg_ratio_shrunk", "Tỷ lệ tiêu cực", {"I3"}),
        ("low_star_ratio_shrunk", "Tỷ lệ 1–2 sao", {"I4"}),
        ("rating_rolling_variance", "Phương sai rating", {"I5"}),
        ("risk_hits", "Lượt khớp từ khóa rủi ro", {"I8"}),
        ("verified_ratio_all", "Tỷ lệ verified", {"I11"}),
    ]
    if features.is_empty():
        st.info("Chưa có feature_series cho SKU này.")
        return
    available = set(features["feature"].unique().to_list())
    for feature, title, indicator_ids in display_features:
        if feature not in available:
            continue
        frame = (
            features.filter(pl.col("feature") == feature)
            .select(pl.col("window_end").alias("date"), "raw_value")
            .drop_nulls()
            .to_pandas()
        )
        chart = _context_chart(
            frame,
            "raw_value",
            title,
            indicator_ids,
            alerts,
            indicators,
            start,
            as_of,
        )
        st.altair_chart(chart, use_container_width=True)
    st.caption(
        "Dữ liệu đang lưu chưa có bảng điểm thay đổi PELT; các chuỗi hiển thị là feature_series "
        "đã được lưu, không nội suy cửa sổ hoặc mức LOW."
    )


def _context_chart(
    frame: Any,
    value_column: str,
    title: str,
    indicator_ids: set[str],
    alerts: pl.DataFrame,
    indicators: pl.DataFrame,
    start: date,
    as_of: date,
) -> alt.LayerChart:
    x = alt.X(
        "date:T",
        title="Ngày",
        scale=alt.Scale(domain=[start.isoformat(), as_of.isoformat()]),
    )
    layers: list[alt.Chart] = []
    if not alerts.is_empty():
        risk = (
            alerts.with_columns(pl.col("window_end").dt.offset_by("1d").alias("end"))
            .select("window_end", "end", "level")
            .to_pandas()
        )
        layers.append(
            alt.Chart(risk)
            .mark_rect(opacity=0.12)
            .encode(
                x=alt.X(
                    "window_end:T",
                    scale=alt.Scale(domain=[start.isoformat(), as_of.isoformat()]),
                ),
                x2="end:T",
                color=alt.Color(
                    "level:N",
                    scale=alt.Scale(
                        domain=["HIGH", "MEDIUM"],
                        range=[LEVEL_COLORS["HIGH"], LEVEL_COLORS["MEDIUM"]],
                    ),
                    legend=None,
                ),
            )
        )
    base = alt.Chart(frame)
    layers.append(
        base.mark_line(color="#3478b5").encode(
            x=x,
            y=alt.Y(f"{value_column}:Q", title=title),
            tooltip=["date:T", f"{value_column}:Q"],
        )
    )
    if indicator_ids and not indicators.is_empty():
        trigger_dates = set(
            indicators.filter(
                pl.col("triggered") & pl.col("indicator_id").is_in(sorted(indicator_ids))
            )["window_end"].to_list()
        )
        points = frame[frame["date"].isin(trigger_dates)]
        if not points.empty:
            layers.append(
                alt.Chart(points)
                .mark_point(color="#742b8e", filled=True, size=60)
                .encode(x=x, y=alt.Y(f"{value_column}:Q"))
            )
    layers.append(
        alt.Chart(pl.DataFrame({"date": [as_of]}).to_pandas())
        .mark_rule(color="#202020", strokeDash=[4, 3])
        .encode(x=x)
    )
    return alt.layer(*layers).resolve_scale(color="independent").properties(height=130, title=title)


def _render_recommendation(result: dict[str, Any]) -> None:
    if result["status"] == "missing":
        st.warning("SKU HIGH này chưa có khuyến nghị. Hãy chạy stage explain.")
        return
    if result["status"] == "invalid":
        st.error(f"File khuyến nghị thiếu hoặc không hợp lệ: {result['error']}")
        return
    recommendation = result["recommendation"]
    if result["is_fallback"]:
        st.info("Khuyến nghị được sinh theo mẫu dự phòng.")
    st.markdown(f"**Tóm tắt:** {recommendation['summary']}")
    st.markdown(
        "**Nguyên nhân sơ bộ:** "
        + " · ".join(cause["cause"] for cause in recommendation["probable_causes"])
    )
    st.markdown(f"**Loại khủng hoảng SCCT:** {recommendation['scct_crisis_type']}")
    columns = st.columns(4)
    for column, (field, title) in zip(columns, DEPARTMENT_LABELS.items(), strict=True):
        with column:
            st.markdown(f"**{title}**")
            for action in recommendation["actions"][field]:
                st.markdown(
                    f"**{action['priority']} · {action['timeline']}**  \n{action['action']}"
                )
    for cause in recommendation["probable_causes"]:
        for review_id in cause["evidence_review_ids"]:
            st.markdown(
                f"[Trích dẫn review `{review_id}`](#review-{escape(review_id, quote=True)})"
            )
    st.caption("Nội dung do LLM tạo, cần xác minh.")


def _verification_tab(category: str, cfg: Any) -> None:
    result = data.validation_summary(cfg, category)
    if result["errors"]:
        for error in result["errors"]:
            st.error(error)
    if result["recall_labels"].is_empty():
        st.info("Chưa có nhãn thu hồi (`recall_labels`) cho category này.")
    else:
        st.dataframe(result["recall_labels"].to_pandas(), hide_index=True, use_container_width=True)
    if not result["reports"]:
        st.info(
            "Chưa có JSON kết quả đánh giá đã tính sẵn trong "
            "`data/processed/evaluation/category=<category>/`; dashboard không tự tính metrics."
        )
    for report in result["reports"]:
        st.markdown(f"**{report['file']}**")
        st.json(report["data"])


def _system_tab(cfg: Any, category: str, alerts: pl.DataFrame, latest_date: date) -> None:
    specs = []
    weights = data.calculate_weights(cfg)
    for indicator_id, spec in cfg.indicators.indicators.items():
        trigger = spec.trigger
        rule = (
            f"{trigger.direction}: z {'>' if trigger.direction == 'up' else '<'} "
            f"{trigger.k} trong {trigger.m}/{trigger.n} cửa sổ"
            if trigger.k is not None
            else f"Luật riêng ({spec.method})"
        )
        specs.append(
            {
                "Chỉ báo": indicator_id,
                "Tên": spec.name,
                "Nhóm": spec.group,
                "Ưu tiên": spec.priority,
                "Trọng số": weights.get(indicator_id, 0.0),
                "Luật kích hoạt": rule,
            }
        )
    st.dataframe(pl.DataFrame(specs).to_pandas(), hide_index=True, use_container_width=True)
    thresholds = thresholds_for_category(category, cfg)
    st.markdown(f"**Ngưỡng:** τ₁ = {thresholds.tau1:.3f} · τ₂ = {thresholds.tau2:.3f}")
    st.markdown(f"**Mốc dữ liệu gần nhất:** {latest_date.isoformat()}")
    if (date.today() - latest_date).days > 7:
        st.warning("Dữ liệu chưa cập nhật: mốc mới nhất đã cũ hơn 7 ngày.")
    st.markdown(
        "**Giới hạn:** Amazon Reviews hiện chỉ có phạm vi SKU; dữ liệu tiếng Anh; "
        "bảng `alerts` chỉ lưu MEDIUM/HIGH; trạng thái `insufficient_data`, kết quả evaluation "
        "và bảng điểm PELT chưa được lưu trong hợp đồng hiện tại."
    )


def main() -> None:
    st.set_page_config(page_title="BrandSentinel · Dashboard", layout="wide")
    st.title("Tổng quan rủi ro sản phẩm")
    cfg = get_config()
    categories = data.available_categories(cfg)
    default_categories = cfg.default.data.categories.dev + cfg.default.data.categories.holdout
    category_options = categories or default_categories
    if not category_options:
        st.info("Chưa có category trong cấu hình hoặc dữ liệu.")
        return

    query_category = st.query_params.get("category")
    initial_category = query_category if query_category in category_options else category_options[0]
    controls = st.columns([2, 3, 1])
    with controls[0]:
        category = st.selectbox(
            "Category", category_options, index=category_options.index(initial_category)
        )
    with controls[2]:
        api_url = (
            st.text_input(
                "API URL",
                value=st.session_state.get("dashboard_api_url", "http://127.0.0.1:8000"),
                placeholder="http://127.0.0.1:8000",
                help="Địa chỉ BrandSentinel API dùng để tạo đề xuất.",
            )
            .strip()
            .rstrip("/")
        )
        st.session_state.dashboard_api_url = api_url

    alerts = _alert_history(category, cfg)
    try:
        daily = data.read_category_table(Table.DAILY_AGG, cfg, category)
        latest_date = daily["date"].max() if not daily.is_empty() else date.today()
    except FileNotFoundError:
        latest_date = alerts["window_end"].max() if not alerts.is_empty() else date.today()

    review_start = max(date.min, latest_date - timedelta(days=13))
    try:
        clean = data.read_category_table(
            Table.CLEAN_REVIEWS, cfg, category, start=review_start, end=latest_date
        )
    except FileNotFoundError:
        clean = pl.DataFrame()
    try:
        nlp = (
            data.read_category_table(
                Table.NLP_FEATURES,
                cfg,
                category,
                start=review_start,
                end=latest_date,
                review_ids=clean["review_id"].to_list(),
            )
            if not clean.is_empty()
            else pl.DataFrame()
        )
    except FileNotFoundError:
        nlp = pl.DataFrame()

    rows, metrics = data.overview_metrics(clean, nlp, alerts, latest_date)
    st.caption(
        f"{category} · 7 ngày gần nhất đến {latest_date.isoformat()} · "
        "Review hợp lệ không gồm review bị gắn cờ spam."
    )
    metric_columns = st.columns(4)
    metric_columns[0].metric("Số đánh giá hợp lệ", f"{metrics['valid_reviews']:,}")
    metric_columns[1].metric("Rủi ro cao", metrics["high_risk"])
    metric_columns[2].metric("Rủi ro trung bình", metrics["medium_risk"])
    negative_pct = metrics["negative_sentiment_pct"]
    metric_columns[3].metric(
        "Tỉ lệ đánh giá tiêu cực",
        f"{negative_pct:.1%}" if negative_pct is not None else "—",
        help="Tính trên review hợp lệ có sentiment_score trong NLP features.",
    )

    if not rows:
        st.info(f"Chưa có review hoặc cảnh báo cho {category}.")
        return

    saved_summaries = (
        _message_block(
            "Không đọc được khuyến nghị đã lưu",
            lambda: data.recommendation_summaries(cfg, category),
        )
        or {}
    )
    api_recommendations = st.session_state.get("dashboard_api_recommendations", {})

    def pct(value: float | None) -> str:
        return f"{value:.1%}" if value is not None else "—"

    def trend_label(value: float | None) -> str:
        if value is None:
            return "Chưa đủ dữ liệu"
        if value > 0.05:
            return f"↑ Tăng {value * 100:.1f} đpt"
        if value < -0.05:
            return f"↓ Giảm {abs(value) * 100:.1f} đpt"
        return "→ Ổn định"

    display_rows = []
    for row in rows:
        window_end = row["window_end"]
        key = f"{category}:{row['sku']}:{window_end.isoformat()}" if window_end else ""
        api_result = api_recommendations.get(key)
        summary = (
            api_result.get("summary")
            if api_result
            else saved_summaries.get((row["sku"], window_end.isoformat()), "")
            if window_end
            else ""
        )
        display_rows.append(
            {
                "SKU / ASIN": row["sku"],
                "Rating TB": f"{row['avg_rating']:.2f}" if row["avg_rating"] is not None else "—",
                "Rating 1–2 sao": pct(row["low_rating_pct"]),
                "Sentiment tiêu cực": pct(row["negative_sentiment_pct"]),
                "Xu hướng tiêu cực": trend_label(row["negative_trend"]),
                "Mức rủi ro": row["level"] or "Chưa cảnh báo",
                "RiskScore": row["score"],
                "Hành động đề xuất": summary or "Chưa có đề xuất",
            }
        )

    selection = st.dataframe(
        display_rows,
        hide_index=True,
        use_container_width=True,
        on_select="rerun",
        selection_mode="single-row",
        key="product_overview_table",
        column_config={
            "RiskScore": st.column_config.ProgressColumn(
                "RiskScore",
                min_value=0.0,
                max_value=1.0,
                format="%.2f",
                help="Điểm ưu tiên kiểm tra, không phải xác suất khủng hoảng.",
            ),
            "Hành động đề xuất": st.column_config.TextColumn("Hành động đề xuất", width="large"),
        },
    )
    selected_rows = selection.selection.rows if selection and selection.selection else []
    selected = rows[selected_rows[0]] if selected_rows else None
    can_request = bool(selected and selected["level"] == "HIGH" and api_url)
    if st.button(
        "Lấy đề xuất từ API",
        disabled=not can_request,
        help="Chọn một dòng HIGH trong bảng để tạo đề xuất." if not can_request else None,
    ):
        assert selected is not None and selected["alert"] is not None
        try:
            pack = data.build_api_evidence_pack(cfg, selected["alert"])
            response = httpx.post(
                f"{api_url}/api/v1/analyze-risk",
                json=pack.model_dump(mode="json"),
                timeout=60,
            )
            response.raise_for_status()
            recommendation = RecommendationOutput.model_validate(response.json())
            if (
                recommendation.sku != selected["sku"]
                or recommendation.category != category
                or recommendation.window_end != selected["window_end"].isoformat()
            ):
                raise ValueError("API trả khuyến nghị không khớp với SKU/cửa sổ đã chọn.")
            result_key = f"{category}:{recommendation.sku}:{recommendation.window_end}"
            st.session_state.setdefault("dashboard_api_recommendations", {})[result_key] = (
                recommendation.model_dump(mode="json")
            )
            st.rerun()
        except (httpx.HTTPError, ValueError) as exc:
            st.error(f"Không thể lấy đề xuất từ API: {exc}")

    if selected and selected["level"] == "HIGH" and selected["window_end"]:
        selected_key = f"{category}:{selected['sku']}:{selected['window_end'].isoformat()}"
        recommendation = api_recommendations.get(selected_key)
        if recommendation:
            _render_recommendation(
                {
                    "status": "ok",
                    "recommendation": recommendation,
                    "is_fallback": recommendation["is_fallback"],
                }
            )
        else:
            saved = data.load_recommendation(cfg, category, selected["sku"], selected["window_end"])
            if saved["status"] == "ok":
                _render_recommendation(saved)

    st.caption(
        "Xu hướng so sánh tỉ lệ sentiment âm 7 ngày hiện tại với 7 ngày trước đó. "
        "Mức “Chưa cảnh báo” nghĩa là chưa có alert MEDIUM/HIGH được lưu."
    )


if __name__ == "__main__":
    main()
