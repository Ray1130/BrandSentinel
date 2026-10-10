import pytest

from brandsentinel.llm.cache import (
    cached_recommendation_path,
    evidence_hash,
    read_cached_recommendation,
    write_cached_recommendation,
)
from brandsentinel.llm.client import require_api_key


def test_evidence_hash_is_stable_for_mapping_order(cfg):
    first = {"sku": "B0EXAMPLE", "reviews": [{"id": "r1", "text": "warning"}]}
    second = {"reviews": [{"text": "warning", "id": "r1"}], "sku": "B0EXAMPLE"}

    assert evidence_hash(first) == evidence_hash(second)
    assert evidence_hash(first) != evidence_hash({**first, "sku": "B0OTHER"})
    assert cached_recommendation_path(cfg, first) == cached_recommendation_path(cfg, second)


def test_evidence_hash_rejects_non_json_numbers():
    with pytest.raises(ValueError, match="Out of range float values"):
        evidence_hash({"score": float("nan")})


def test_recommendation_is_cached_by_evidence_hash(cfg, tmp_path):
    isolated_cfg = cfg.model_copy(update={"root": tmp_path})
    evidence = {"sku": "B0EXAMPLE", "reviews": [{"id": "r1"}]}
    recommendation = {"summary": "Recall risk; verify source"}

    assert read_cached_recommendation(isolated_cfg, evidence) is None
    write_cached_recommendation(isolated_cfg, evidence, recommendation)

    assert read_cached_recommendation(isolated_cfg, evidence) == recommendation


def test_openai_key_is_required_only_when_requested(cfg, monkeypatch):
    monkeypatch.delenv(cfg.llm.api_key_env, raising=False)

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        require_api_key(cfg.llm)
