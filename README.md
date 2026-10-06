# BrandSentinel

Hệ thống cảnh báo sớm khủng hoảng chất lượng sản phẩm từ review (STL + PELT + 11 chỉ báo + LLM diễn giải).
Thiết kế chi tiết: `docs/data_contract.md`.

## Cài đặt (Windows, PowerShell)

```powershell
# 1. Cài uv (một lần)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# 2. Clone và cài môi trường
git clone <repo-url> brandsentinel
cd brandsentinel
uv sync                      # lõi + dev (không cần torch)
# uv sync --extra nlp        # chỉ khi làm NLP (P3): nặng, cài torch
# uv sync --extra serving    # chỉ khi làm API/dashboard

# 3. Kiểm tra
uv run poe check             # lint + format-check + test
```

Trên Linux/macOS các lệnh `uv` giống hệt. **Không dùng `make`** (không có sẵn trên Windows).

## Lệnh thường dùng

| Việc | Lệnh |
|---|---|
| Chạy test | `uv run poe test` |
| Lint / format | `uv run poe lint` / `uv run poe fmt` |
| Lint + format-check + test | `uv run poe check` |
| Tạo cây file rỗng theo thiết kế | `uv run poe scaffold` |
| Chạy một stage | `uv run bs run --stage preprocess --category Baby_Products --start 2023-01-01 --end 2023-03-31` |

## Phân công thư mục

| Người | Thư mục sở hữu | Task đầu tiên |
|---|---|---|
| P1 | `core/`, `preprocessing/`, `features/{windows,volume,rating,reliability}.py`, `pipeline.py`, `cli.py`, `serving/` | T1 |
| P2 | `detection/`, `scoring/`, `evaluation/{synthetic,metrics,baselines}.py` | T3 |
| P3 | `features/nlp/`, `detection/indicators/content.py`, `llm/`, `configs/lexicon_*.yaml` | T5 |

`core/` và `docs/data_contract.md` chỉ sửa qua PR có người làm T0 review.

## Quy ước làm việc

- Nhánh: `main` được bảo vệ; làm trên nhánh `p<số>/<task>` (ví dụ `p1/t1-loader`); PR cần 1 review chéo.
- Không commit `data/` (đã ignore). Dữ liệu thật tải bằng `scripts/download_amazon.py`.
- Mọi file text đọc/ghi với `encoding="utf-8"`.
- Mỗi người phát triển trên dữ liệu mock cho đến khi bảng thật sẵn sàng: `from brandsentinel.testing.mock_data import make_all` (đủ 6 bảng, đúng hợp đồng). Fixture pytest `mock_tables` và `cfg` có sẵn trong `tests/conftest.py`.
- Mọi bảng đọc/ghi qua `core/io.py` (`read_table`, `write_table`) và kiểm bằng `core/schemas.validate` trước khi ghi.
- Chỉ báo mới: kế thừa `detection/base.Indicator`, gắn `@register` (xem `core/registry.py`).
