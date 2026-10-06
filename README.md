# BrandSentinel

**Phát hiện sớm dấu hiệu khủng hoảng chất lượng sản phẩm từ review trực tuyến.**
Dự án của đội TrainToOlympus tại AISC'26.

BrandSentinel là hệ thống cảnh báo sớm (Early Warning System) ở cấp từng mã sản phẩm (SKU). Hệ thống
theo dõi review theo thời gian, phát hiện tín hiệu bất thường và báo cho doanh nghiệp *trước khi*
sự cố bùng nổ trên truyền thông, kèm bằng chứng và gợi ý hành động cho từng bộ phận.

## Ý tưởng cốt lõi

- **Từ phản ứng sang phòng ngừa.** Điểm rating trung bình cộng dồn hay doanh số phản ứng chậm với sản
  phẩm đã có nhiều review. Thay vào đó, hệ thống theo dõi *sự thay đổi bất thường* của nhiều chỉ báo
  theo thời gian.
- **Không cần dữ liệu gán nhãn.** Không có nhãn "khủng hoảng / không khủng hoảng" để huấn luyện. Hệ thống
  đo độ lệch của từng SKU so với baseline lịch sử của chính nó và của ngành hàng. Trọng số và ngưỡng đặt
  theo quy tắc nghiệp vụ, công khai trong file cấu hình.
- **Minh bạch, giải thích được.** Mỗi cảnh báo chỉ rõ chỉ báo nào được kích hoạt và kèm review làm bằng
  chứng. Điểm rủi ro dùng để **xếp hạng ưu tiên kiểm tra**, không phải xác suất khủng hoảng; mức HIGH
  nghĩa là "cần xác minh ngay", chưa phải kết luận khủng hoảng.
- **Cảnh báo chỉ khi có xác nhận chéo.** Một tín hiệu đơn lẻ (ví dụ review tăng vì khuyến mãi) không đủ
  để báo động. Muốn lên mức cao, tín hiệu phải được duy trì theo thời gian và được xác nhận bởi nhiều
  nhóm tín hiệu khác nhau.

## Cách hoạt động

```
Review thô ──► Tiền xử lý ──► Đặc trưng ──► Phát hiện bất thường ──► Chấm điểm ──► LLM diễn giải
 (SKU, ts,      loại trùng,    volume,       khử mùa vụ (STL),       RiskScore,     tóm tắt, nguyên
 rating, text,  gắn cờ spam,   rating,       xu hướng, điểm thay     LOW/MEDIUM/    nhân sơ bộ, hành
 verified)      chuẩn hóa,     NLP           đổi (PELT) → 11 chỉ     HIGH + cờ      động theo phòng
                gom SKU×ngày                 báo                     xác minh       ban (chỉ với HIGH)
```

| Bước | Làm gì | Thuật toán chính |
|---|---|---|
| **Tiền xử lý** | Loại trùng, gắn cờ spam/bot (không xóa), chuẩn hóa văn bản, gom theo SKU × ngày | Hash dedup, luật spam, chuẩn hóa Unicode/emoji/teencode |
| **Đặc trưng** | Biến review thành chuỗi đặc trưng theo cửa sổ trượt 7 ngày (so với baseline 90 ngày) | Thống kê cửa sổ trượt, co tỷ lệ về trung bình ngành (Beta), sentiment, từ điển rủi ro, khía cạnh |
| **Phát hiện** | Loại biến động thông thường, tìm thay đổi bất thường | Median/MAD và robust z-score, **STL**, **Mann–Kendall**, **PELT** (`ruptures`) |
| **Chấm điểm** | Tổng hợp chỉ báo thành điểm và mức rủi ro | Tổng có trọng số + luật xác nhận chéo + persistence |
| **Diễn giải** | Giải thích vì sao SKU bị cảnh báo và nên làm gì | LLM theo khung lý thuyết truyền thông khủng hoảng SCCT |

### 11 chỉ báo rủi ro

| Nhóm | Chỉ báo | Ý nghĩa |
|---|---|---|
| **Tần suất, diễn biến** | I1 Đột biến volume sau khử mùa vụ | Review tăng bất thường, đã trừ chu kỳ tuần và khuyến mãi |
| | I2 Tốc độ tăng review bất thường | Hữu ích cho sản phẩm mới, chưa đủ lịch sử cho STL |
| | I3 Xu hướng tiêu cực kéo dài | Tỷ lệ review xấu nhích dần qua từng tuần |
| **Rating, trải nghiệm** | I4 Gia tăng tỷ lệ 1–2 sao | Trải nghiệm đang xấu đi |
| | I5 Biến động rating kéo dài | Trải nghiệm đang bất ổn (ví dụ lô hàng không đồng đều) |
| | I6 Phân cực 1 sao và 5 sao | Dấu hiệu lô lỗi lẫn lô tốt, hoặc review giả kéo điểm |
| **Nội dung, ngữ nghĩa** | I7 Dịch chuyển chủ đề sang vấn đề mới | Xuất hiện cụm phàn nàn mới |
| | **I8 An toàn, sức khỏe, pháp lý** | Ưu tiên rất cao: có thể dẫn đến nghĩa vụ thu hồi |
| | I9 Tiêu cực theo khía cạnh | Mức nặng của vấn đề (chất lượng, giao hàng, an toàn, hoàn tiền) |
| **Cờ độ tin cậy** (không vào điểm) | I10 Rating và nội dung lệch pha | Review mỉa mai hoặc bấm nhầm sao |
| | I11 Tỷ lệ verified bất thường | Dấu hiệu review bombing khi volume tăng mà verified giảm |

### Phân mức rủi ro

`RiskScore = Σ wᵢ · Iᵢ` (Iᵢ ∈ {0, 1}, chỉ I1–I9; trọng số suy ra từ mức ưu tiên nghiệp vụ).

| Mức | Điều kiện |
|---|---|
| **LOW** | Điểm < τ₁ |
| **MEDIUM** | Điểm ≥ τ₁ và (tín hiệu duy trì ≥ 2 cửa sổ liên tiếp, hoặc ≥ 2/3 nhóm tín hiệu xác nhận) |
| **HIGH** | Điểm ≥ τ₂ và (đủ cả 3 nhóm tín hiệu, hoặc có bằng chứng an toàn nghiêm trọng từ I8) |

Đủ điểm nhưng thiếu xác nhận thì hạ xuống mức liền kề thấp hơn. Nếu I10 hoặc I11 báo bất thường, SKU được
gắn thêm cờ **"cần xác minh tính xác thực"**, không phải một mức riêng. Ngưỡng τ₁, τ₂ hiệu chỉnh theo
phân phối nền của từng ngành hàng.

## Thiết kế mã nguồn

- **Mô-đun hóa theo giai đoạn**, mỗi giai đoạn đọc và ghi bảng Parquet nên chạy lại độc lập được.
- **Hợp đồng dữ liệu** gồm 6 bảng (`clean_reviews`, `daily_agg`, `nlp_features`, `feature_series`,
  `indicator_matrix`, `alerts`), được kiểm tự động bằng Pandera ở `core/schemas.py`.
  Mô tả chi tiết: [`docs/data_contract.md`](docs/data_contract.md).
- **Chỉ báo cắm được:** mỗi chỉ báo kế thừa `detection/base.Indicator` và đăng ký bằng `@register`;
  bật/tắt, trọng số và tham số nằm ở `configs/indicators.yaml`.
- **Không hard-code tham số:** mọi ngưỡng, cửa sổ, từ điển đều ở `configs/` (YAML, được kiểm bằng Pydantic).
  Các giá trị hiện tại là cấu hình khởi tạo, được hiệu chỉnh trên dữ liệu baseline.
- **LLM chỉ diễn giải:** chỉ nhận "gói bằng chứng", trả JSON theo schema, phải trích dẫn review có thật
  và không được tự đổi mức rủi ro.

```
src/brandsentinel/
├── core/            # config, schemas, types, ids, io, logging, registry
├── preprocessing/   # loader, validate, dedup, spam, normalize, aggregate
├── features/        # cửa sổ, volume, rating, độ tin cậy; nlp/ (sentiment, từ điển rủi ro, khía cạnh)
├── detection/       # baseline, STL, xu hướng, PELT, kích hoạt; indicators/ (I1–I11)
├── scoring/         # trọng số, nhóm tín hiệu, RiskScore, ngưỡng, luật phân mức, trạng thái
├── llm/             # gói bằng chứng, prompt, kiểm tra đầu ra, dự phòng
├── evaluation/      # ghép với dữ liệu thu hồi, chỉ số, baseline so sánh, dữ liệu tổng hợp
├── serving/         # API, dashboard, thông báo
├── testing/         # dữ liệu mock đúng hợp đồng cho cả 6 bảng
├── pipeline.py      # nối các giai đoạn
└── cli.py           # lệnh `bs`
configs/             # default.yaml, indicators.yaml, thresholds.yaml, lexicon_*.yaml, llm.yaml
docs/                # data_contract.md
```

## Dữ liệu và đánh giá

- **Dữ liệu:** Amazon Reviews 2023 (tiếng Anh, khóa sản phẩm `parent_asin`, có `verified_purchase`).
- **Kiểm chứng không cần nhãn:** đối chiếu với danh sách thu hồi sản phẩm công khai (lead time =
  ngày thu hồi trừ ngày xuất hiện HIGH đầu tiên), so với baseline đơn giản (rating trượt, chỉ volume, chỉ
  từ khóa), ablation theo nhóm tín hiệu, và chèn khủng hoảng tổng hợp để hiệu chỉnh ngưỡng.

## Hạn chế đã biết

- Dataset lấy mẫu theo người dùng nên độ phủ review của một sản phẩm có thể không đầy đủ, ảnh hưởng các
  chỉ báo dựa trên volume (I1, I2).
- Prototype kiểm chứng trên dữ liệu tiếng Anh/thị trường Mỹ; mở rộng sang tiếng Việt (PhoBERT, sàn nội
  địa) là hướng phát triển.
- STL và PELT cần SKU có đủ lịch sử và đủ review mỗi tuần; SKU quá thưa bị đánh dấu `insufficient_data`.
- PELT là thuật toán offline nên có độ trễ phát hiện tối thiểu bằng kích thước đoạn nhỏ nhất.

## Hướng mở rộng

Xử lý luồng thời gian thực (Kafka, Flink) · NLP tiếng Việt (PhoBERT, từ điển phương ngữ/từ lóng) ·
BERTopic cho I7 · webhook vào Slack, Telegram, Lark, Jira, Zendesk.

## Bắt đầu nhanh (Windows, PowerShell)

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"   # cài uv (một lần)
git clone <repo-url> brandsentinel; cd brandsentinel
uv sync                    # lõi + dev; thêm --extra nlp cho NLP (cài torch), --extra serving cho API/dashboard
uv run poe check           # lint + format-check + test
```

| Việc | Lệnh |
|---|---|
| Test / lint / format | `uv run poe test` / `uv run poe lint` / `uv run poe fmt` |
| Chạy một giai đoạn | `uv run bs run --stage preprocess --category Baby_Products --start 2023-01-01 --end 2023-03-31` |
| Dữ liệu mock | `from brandsentinel.testing.mock_data import make_all` |

Không dùng `make` (không có sẵn trên Windows). Mọi file text đọc/ghi bằng UTF-8.