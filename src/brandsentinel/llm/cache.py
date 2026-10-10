"""Stable cache keys for LLM recommendations."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from brandsentinel.core.config import Config
from brandsentinel.core.io import read_json, recommendation_cache_path, write_json


def evidence_hash(evidence: Mapping[str, Any]) -> str:
    """Hash a JSON evidence package independent of mapping insertion order."""
    canonical = json.dumps(
        evidence,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def cached_recommendation_path(cfg: Config, evidence: Mapping[str, Any]) -> Path:
    """Locate a cached recommendation by the canonical evidence hash."""
    return recommendation_cache_path(cfg, evidence_hash(evidence))


def read_cached_recommendation(cfg: Config, evidence: Mapping[str, Any]) -> Any | None:
    """Return a cached JSON result, or None when the evidence has not been evaluated."""
    path = cached_recommendation_path(cfg, evidence)
    return read_json(path) if path.is_file() else None


def write_cached_recommendation(
    cfg: Config, evidence: Mapping[str, Any], recommendation: Any
) -> Path:
    """Persist a result so an identical evidence package does not need another LLM call."""
    return write_json(cached_recommendation_path(cfg, evidence), recommendation)
