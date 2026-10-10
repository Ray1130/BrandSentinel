# Hợp đồng dữ liệu

Các bảng được kiểm tra bằng Pandera trong `src/brandsentinel/core/schemas.py`. Mỗi khóa chính phải
duy nhất trong bảng; Parquet được phân vùng theo `category` và tháng của cột ngày được liệt kê dưới
đây.

| Bảng | Khóa | Cột phân vùng ngày | Vùng |
|---|---|---|---|
| `clean_reviews` | `review_id` | `date` | `data_interim` |
| `daily_agg` | `sku`, `date` | `date` | `data_interim` |
| `nlp_features` | `review_id` | `date` | `data_interim` |
| `feature_series` | `sku`, `window_end`, `indicator_id`, `feature` | `window_end` | `data_processed` |
| `indicator_matrix` | `sku`, `window_end`, `indicator_id` | `window_end` | `data_processed` |
| `alerts` | `sku`, `window_end` | `window_end` | `data_processed` |
| `recall_labels` | `sku`, `recall_id` | `recall_date` | `data_interim` |

## `recall_labels`

Đây là bảng nhãn kiểm chứng bên ngoài, tách khỏi các bảng tính toán của pipeline. Nguồn MVP là CPSC;
SKU phải là ASIN trong Amazon Reviews 2023 `parent_asin`, và chỉ nhận kết nối ASIN chính xác. Không
được gán nhãn bằng cách chỉ so gần đúng tên sản phẩm.

| Cột | Kiểu | Nullable | Ý nghĩa |
|---|---|---|---|
| `sku` | string | Không | ASIN (`parent_asin`) khớp chính xác |
| `category` | string | Không | Category Amazon, ví dụ `Baby_Products` |
| `recall_id` | string | Không | Mã thu hồi của nguồn |
| `recall_date` | date | Không | Ngày thu hồi công bố |
| `source` | string | Không | Nhà cung cấp danh sách, MVP: `CPSC` |
| `source_url` | string | Không | URL HTTPS dẫn tới hồ sơ nguồn |
| `product_name` | string | Không | Tên sản phẩm trong hồ sơ thu hồi |
| `match_type` | string | Không | Phải là `exact_asin` |
| `match_confidence` | float | Không | Độ tin cậy định danh; exact ASIN = `1.0`, nằm trong `[0, 1]` |

Khóa `(sku, recall_id)` cho phép một ASIN có nhiều lần thu hồi nhưng ngăn trùng cùng một sự kiện.
`source_url` phải dùng HTTPS. Đầu vào ghép phải có ASIN đã được kiểm tay (`manually_verified=true`)
và URL hồ sơ làm bằng chứng; chỉ bản ghi đã kiểm tay, khớp chính xác với `clean_reviews.sku` mới
được xuất thành nhãn. `match_confidence` mô tả độ khớp định danh, không thay thế việc kiểm tay.
Không coi nhãn thiếu là nhãn âm.

## `alerts`

`alerts` chỉ lưu cửa sổ đã đạt mức `MEDIUM` hoặc `HIGH`. Cửa sổ `LOW` vẫn có thể được xem trong
`indicator_matrix`/`feature_series`, nhưng không nằm trong mẫu số precision của cảnh báo gửi người
dùng. Cờ `verify_flag` (bao gồm I11/review bombing) là metadata để kiểm tra tính xác thực; cờ này
không tự hạ điểm hoặc giới hạn mức rủi ro.

## Cache và cấu hình LLM

Thiết lập nhà cung cấp nằm trong `configs/llm.yaml`. Khóa API chỉ được đọc từ biến môi trường được
khai báo ở `api_key_env`; không lưu khóa trong YAML, dữ liệu hoặc repo. Cache khuyến nghị dùng
SHA-256 của JSON chuẩn hóa của toàn bộ gói bằng chứng (`brandsentinel.llm.cache.evidence_hash`), để
thứ tự khóa JSON không làm phát sinh lần gọi mới. Ngân sách cấu hình ban đầu là 5 USD/tháng.
