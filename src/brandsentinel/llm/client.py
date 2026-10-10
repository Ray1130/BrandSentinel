"""Credential access for configured LLM providers."""

from __future__ import annotations

import os

from brandsentinel.core.config import LlmConfig


def require_api_key(config: LlmConfig) -> str:
    """Read the configured secret from the environment without logging or persisting it."""
    key = os.environ.get(config.api_key_env)
    if not key:
        raise RuntimeError(
            f"LLM API key is not set; provide it via the {config.api_key_env} environment variable"
        )
    return key
