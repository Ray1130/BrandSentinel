"""Logging và audit log (số bản ghi bị loại ở mỗi bước tiền xử lý)."""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

_ROOT = "brandsentinel"


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Gọi một lần ở điểm vào (cli.py). Gọi lại không nhân đôi handler."""
    logger = logging.getLogger(_ROOT)
    logger.setLevel(level.upper())
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
        logger.addHandler(handler)
    return logger


def get_logger(name: str) -> logging.Logger:
    """Logger con của `brandsentinel`, ví dụ get_logger(__name__)."""
    return logging.getLogger(name if name.startswith(_ROOT) else f"{_ROOT}.{name}")


class AuditLog:
    """Ghi lại số bản ghi vào/ra của từng bước, ví dụ dedup, spam, normalize."""

    def __init__(self, stage: str):
        self.stage = stage
        self.steps: list[dict[str, Any]] = []

    def record(self, step: str, n_in: int, n_out: int, **extra: Any) -> None:
        self.steps.append(
            {"step": step, "n_in": n_in, "n_out": n_out, "removed": n_in - n_out, **extra}
        )

    def to_dataframe(self) -> pl.DataFrame:
        if not self.steps:
            return pl.DataFrame(schema={"step": pl.String, "n_in": pl.Int64, "n_out": pl.Int64})
        return pl.DataFrame(self.steps)

    def write(self, directory: Path) -> Path:
        """Nối thêm vào <directory>/<stage>.jsonl (mỗi dòng một bước, kèm thời điểm chạy)."""
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.stage}.jsonl"
        now = datetime.now(UTC).isoformat(timespec="seconds")
        with path.open("a", encoding="utf-8") as f:
            for s in self.steps:
                f.write(
                    json.dumps({"ts": now, "stage": self.stage, **s}, ensure_ascii=False) + "\n"
                )
        return path
