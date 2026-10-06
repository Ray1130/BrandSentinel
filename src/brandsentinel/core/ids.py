"""Sinh khóa ổn định. Chạy lại pipeline không làm đổi ID."""

from __future__ import annotations

import hashlib


def review_id(parent_asin: str, user_id: str | None, timestamp_ms: int) -> str:
    """sha1(parent_asin|user_id|timestamp_ms), lấy 16 ký tự hex đầu. user_id null -> chuỗi rỗng."""
    raw = f"{parent_asin}|{user_id or ''}|{int(timestamp_ms)}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
