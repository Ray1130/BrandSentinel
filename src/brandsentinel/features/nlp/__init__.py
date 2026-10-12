"""NLP feature generation modules and pipeline orchestration."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from brandsentinel.features.nlp.aspect import (
        AspectExtractor,
        compute_aspect,
        compute_aspect_window_features,
        get_aspect_extractor,
    )
    from brandsentinel.features.nlp.embedder import (
        RiskEmbedder,
        compute_embedding_similarity,
        compute_i8_combined,
        get_risk_embedder,
    )
    from brandsentinel.features.nlp.pipeline import compute_nlp_features
    from brandsentinel.features.nlp.risk_lexicon import (
        RiskLexiconDetector,
        compute_risk_lexicon,
        compute_risk_window_features,
        get_risk_detector,
    )
    from brandsentinel.features.nlp.sentiment import (
        SentimentAnalyzer,
        compute_sentiment,
        get_sentiment_analyzer,
    )
    from brandsentinel.features.nlp.topic import (
        compute_term_burst,
        compute_topic_window_features,
        extract_ngrams,
    )

_EXPORT_MODULES = {
    "AspectExtractor": "aspect",
    "compute_aspect": "aspect",
    "compute_aspect_window_features": "aspect",
    "get_aspect_extractor": "aspect",
    "RiskEmbedder": "embedder",
    "compute_embedding_similarity": "embedder",
    "compute_i8_combined": "embedder",
    "get_risk_embedder": "embedder",
    "compute_nlp_features": "pipeline",
    "RiskLexiconDetector": "risk_lexicon",
    "compute_risk_lexicon": "risk_lexicon",
    "compute_risk_window_features": "risk_lexicon",
    "get_risk_detector": "risk_lexicon",
    "SentimentAnalyzer": "sentiment",
    "compute_sentiment": "sentiment",
    "get_sentiment_analyzer": "sentiment",
    "compute_term_burst": "topic",
    "compute_topic_window_features": "topic",
    "extract_ngrams": "topic",
}


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.{module_name}"), name)
    globals()[name] = value
    return value


__all__ = [
    "AspectExtractor",
    "RiskEmbedder",
    "RiskLexiconDetector",
    "SentimentAnalyzer",
    "compute_aspect",
    "compute_aspect_window_features",
    "compute_embedding_similarity",
    "compute_i8_combined",
    "compute_nlp_features",
    "compute_risk_lexicon",
    "compute_risk_window_features",
    "compute_sentiment",
    "compute_term_burst",
    "compute_topic_window_features",
    "extract_ngrams",
    "get_aspect_extractor",
    "get_risk_detector",
    "get_risk_embedder",
    "get_sentiment_analyzer",
]
