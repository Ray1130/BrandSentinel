"""NLP feature generation modules: sentiment, risk detection, aspect extraction, semantic embeddings, and pipeline orchestration."""

from __future__ import annotations

from brandsentinel.features.nlp.aspect import (
    AspectExtractor,
    compute_aspect,
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
    get_risk_detector,
)
from brandsentinel.features.nlp.sentiment import (
    SentimentAnalyzer,
    compute_sentiment,
    get_sentiment_analyzer,
)

__all__ = [
    "AspectExtractor",
    "RiskEmbedder",
    "RiskLexiconDetector",
    "SentimentAnalyzer",
    "compute_aspect",
    "compute_embedding_similarity",
    "compute_i8_combined",
    "compute_nlp_features",
    "compute_risk_lexicon",
    "compute_sentiment",
    "get_aspect_extractor",
    "get_risk_detector",
    "get_risk_embedder",
    "get_sentiment_analyzer",
]
