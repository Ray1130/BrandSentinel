# Hợp đồng dữ liệu BrandSentinel

> **Trạng thái:** BẢN NHÁP — người phụ trách T0 chốt và khóa trước khi P2, P3 clone repo.
> Nguồn sự thật là code: `core/schemas.py` (kiểu, null, khóa, luật ngữ nghĩa) và `core/types.py`
> (tên đặc trưng). File này là bản mô tả; nếu lệch thì sửa cho khớp code.
> Mọi thay đổi sau khi khóa phải đi qua PR sửa file này **và** `src/brandsentinel/core/schemas.py`.
> Các mục đánh dấu `CHỐT:` là chỗ cần điền/xác nhận.

## 1. Quy ước chung

| Hạng mục | Quy ước | Ghi chú |
|---|---|---|
| Khóa SKU | `sku = parent_asin` (Amazon Reviews 2023) | CHỐT: xác nhận không dùng `asin` |
| Khóa review | `review_id = core.ids.review_id(parent_asin, user_id, timestamp_ms)` = 16 ký tự hex đầu của sha1(`parent_asin\|user_id\|timestamp_ms`), user_id null thành chuỗi rỗng | Chạy lại không đổi ID |
| Thời gian | Lưu UTC. `ts` là `Datetime[UTC]`, `date` là ngày UTC của `ts` | Không dùng giờ địa phương ở bất kỳ bảng nào |
| Cửa sổ | `window_end` = ngày cuối **đã bao gồm** của cửa sổ W ngày (W = 7), bước 1 ngày | Cửa sổ gồm `[window_end - W + 1, window_end]` |
| Ngày không có review | `n = 0`; các tỷ lệ và phương sai là `null` (**không phải 0**) | STL/PELT không được nhận số 0 giả |
| Mã chỉ báo | `"I1"` … `"I11"`; nhóm: `volume`, `rating`, `content`, `flag` | I10, I11 thuộc nhóm `flag`, không cộng vào RiskScore |
| Đặt tên | `snake_case`; chuỗi là `Utf8`, đếm là `Int32`, tỷ lệ là `Float64` | CHỐT: có dùng `Categorical` cho `category`/`level` không |
| Lưu trữ | Parquet (zstd): `<zone>/<table>/category=<category>/<yyyy-mm>.parquet`; `interim`: clean_reviews, daily_agg, nlp_features; `processed`: feature_series, indicator_matrix, alerts | Đọc/ghi qua `core/io.py`; tên category chỉ gồm `A-Z a-z 0-9 _ . -` (Windows cấm `: / \`) |
| Idempotent | `write_table` thay các dòng **cùng SKU nằm trong khoảng ngày [min, max] của dữ liệu mới** (nlp_features: thay theo khóa), giữ nguyên dòng khác | Chạy lại không nhân đôi dữ liệu; chạy lại một phần SKU không xóa SKU khác |
| Mã hóa file | Mọi `open()`/đọc ghi text dùng `encoding="utf-8"` | Máy Windows mặc định không phải UTF-8 |
| NaN/inf | Không dùng; giá trị không xác định là `null` | `validate` từ chối NaN/inf |
| Seed | Một seed duy nhất trong `configs/default.yaml` | Dùng cho sampling, MinHash, BERTopic |
| Spam | Chỉ **gắn cờ** `is_spam`, không xóa khỏi `clean_reviews` | I10, I11 tính trên dữ liệu chưa lọc spam; I1–I9 trên dữ liệu đã lọc |

## 2. Luồng stage → bảng

| Stage (`bs run --stage`) | Đọc | Ghi | Chủ sở hữu |
|---|---|---|---|
| `preprocess` | `data/raw/*.jsonl` | `clean_reviews`, `daily_agg` | P1 |
| `features` | `clean_reviews`, `daily_agg`, `nlp_features` | `feature_series` | P1 (không-NLP), P3 (NLP) |
| `detect` | `feature_series` | `indicator_matrix` | P2 |
| `score` | `indicator_matrix` | `alerts` | P2 |
| `explain` | `alerts` (HIGH), `clean_reviews` | `recommendations.json` | P3 |
| `evaluate` | `alerts`, danh sách thu hồi | báo cáo metric | P1, P2 |

Bảng NLP (`nlp_features`) do P3 ghi từ `clean_reviews` trong stage `features`.
Dữ liệu mock cho cả 6 bảng: `from brandsentinel.testing.mock_data import make_all`.

## 3. Schema các bảng (bản nháp để chốt)

### 3.1 `clean_reviews` — ghi: P1, đọc: P1, P3

| Cột | Kiểu | Null | Mô tả |
|---|---|---|---|
| `review_id` | Utf8 | không | Khóa chính |
| `sku` | Utf8 | không | `parent_asin` |
| `category` | Utf8 | không | Category Amazon (dùng cho baseline ngành) |
| `user_id` | Utf8 | có | Dùng cho luật spam và đếm user phân biệt (I8) |
| `ts` | Datetime[UTC] | không | Thời điểm đăng |
| `date` | Date | không | Ngày UTC của `ts` |
| `rating` | Int8 | không | 1–5 |
| `text_raw` | Utf8 | có | Văn bản gốc |
| `text_norm` | Utf8 | có | Văn bản đã chuẩn hóa |
| `verified_purchase` | Boolean | có | |
| `is_spam` | Boolean | không | Cờ spam/bot (không xóa dòng) |
| `spam_reason` | Utf8 | có | Luật nào kích hoạt cờ |

### 3.2 `daily_agg` — ghi: P1, đọc: P1

Một dòng cho mỗi `(sku, date)`, **điền 0 cho ngày thiếu**.

| Cột | Kiểu | Null | Mô tả |
|---|---|---|---|
| `sku`, `category`, `date` | Utf8, Utf8, Date | không | Khóa |
| `n` | Int32 | không | Số review **sau khi lọc spam** |
| `n_neg` | Int32 | không | Số review rating ≤ 2 (sau lọc spam) |
| `n1` … `n5` | Int32 | không | Histogram sao (sau lọc spam) |
| `sum_rating`, `sumsq_rating` | Float64 | không | Để tính phương sai trượt |
| `n_verified` | Int32 | không | Số verified (sau lọc spam) |
| `n_all`, `n_verified_all` | Int32 | không | Như trên nhưng **trước lọc spam** (cho I11) |

### 3.3 `nlp_features` — ghi: P3, đọc: P1, P2

| Cột | Kiểu | Null | Mô tả |
|---|---|---|---|
| `review_id` | Utf8 | không | Khóa |
| `category`, `date` | Utf8, Date | không | Sao chép từ `clean_reviews` (dùng để phân vùng) |
| `sentiment_score` | Float32 | có | Trong [-1, 1] |
| `risk_hit` | Boolean | không | Có nội dung an toàn/sức khỏe/pháp lý (I8) |
| `risk_terms` | List[Utf8] | có | Từ/cụm khớp |
| `risk_sim` | Float32 | có | Độ tương đồng embedding tầng hai |
| `aspect` | Utf8 | có | Một khía cạnh chính/review, thuộc `quality`, `delivery`, `safety`, `refund`. CHỐT: nếu cần nhiều khía cạnh/review thì đổi sang bảng riêng theo (review, aspect) |
| `aspect_sentiment` | Float32 | có | Sentiment theo khía cạnh (I9) |
| `mismatch` | Boolean | có | Rating lệch với sentiment văn bản (I10) |

### 3.4 `feature_series` (dạng long) — ghi: P1, P3, đọc: P2

| Cột | Kiểu | Null | Mô tả |
|---|---|---|---|
| `sku`, `category`, `window_end` | Utf8, Utf8, Date | không | Khóa |
| `indicator_id` | Utf8 | không | `"I1"`…`"I11"` |
| `feature` | Utf8 | không | Tên chuỗi đặc trưng trong chỉ báo (bảng bên dưới); khóa duy nhất là (`sku`, `window_end`, `indicator_id`, `feature`) |
| `raw_value` | Float64 | có | Giá trị đặc trưng thô; null khi không xác định |
| `n_window` | Int32 | không | Số review trong cửa sổ (cổng dữ liệu tối thiểu) |

Cặp (`indicator_id`, `feature`) hợp lệ — khai báo trong `core/types.py::FEATURES_BY_INDICATOR`,
thêm đặc trưng mới phải sửa ở đó và ở bảng này (qua PR). Đặc trưng cấp ngày
(`log1p_daily_count`, `growth_rate`) dùng `window_end` = chính ngày đó.

| Chỉ báo | `feature` |
|---|---|
| I1 | `log1p_daily_count` |
| I2 | `growth_rate` |
| I3 | `neg_ratio_shrunk` |
| I4 | `low_star_ratio_shrunk` |
| I5 | `rating_rolling_variance` |
| I6 | `extreme_share`, `share_1star`, `share_5star` |
| I7 | `term_burst_score` |
| I8 | `risk_hits`, `risk_distinct_users` |
| I9 | `aspect_neg_quality`, `aspect_neg_delivery`, `aspect_neg_safety`, `aspect_neg_refund` |
| I10 | `mismatch_rate` |
| I11 | `verified_ratio_all` |

### 3.5 `indicator_matrix` — ghi: P2, đọc: P2

| Cột | Kiểu | Null | Mô tả |
|---|---|---|---|
| `sku`, `category`, `window_end` | Utf8, Utf8, Date | không | Khóa |
| `indicator_id` | Utf8 | không | |
| `triggered` | Boolean | không | I_i ∈ {0,1} |
| `strength` | Float64 | có | Độ mạnh (robust z hoặc tương đương) để giải thích |

### 3.6 `alerts` — ghi: P2, đọc: P3, P1

| Cột | Kiểu | Null | Mô tả |
|---|---|---|---|
| `sku`, `category`, `window_end` | Utf8, Utf8, Date | không | Khóa |
| `score` | Float64 | không | RiskScore (chỉ I1–I9) |
| `level` | Utf8 | không | `LOW` / `MEDIUM` / `HIGH` |
| `groups_active` | List[Utf8] | không | Nhóm tín hiệu đang active |
| `persist` | Int8 | không | Số cửa sổ liên tiếp duy trì |
| `verify_flag` | Boolean | không | Cờ "cần xác minh tính xác thực" (I10/I11) |
| `triggered_ids` | List[Utf8] | không | Danh sách chỉ báo đã kích hoạt |

### 3.7 `recommendations.json` — ghi: P3

Schema chính thức là `src/brandsentinel/llm/schemas.py` (Pydantic): `summary`,
`probable_causes[{cause, evidence_review_ids, confidence}]`, `scct_crisis_type`,
`actions{QA_KyThuat, VanHanh, CSKH, TruyenThong}`. Mọi `review_id` được trích phải có trong gói
bằng chứng.

## 4. Quy trình đổi hợp đồng

1. Mở PR sửa file này và `core/schemas.py` cùng lúc, nêu rõ bảng/cột bị ảnh hưởng.
2. Cập nhật hàm mock trong `tests/fixtures/` để vẫn qua schema.
3. Người ghi và người đọc của bảng đó đều phải approve.
4. Sau khi merge, báo cả nhóm trong buổi họp ngắn hằng ngày.
