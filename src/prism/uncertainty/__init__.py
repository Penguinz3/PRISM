"""Uncertainty scoring interfaces for PRISM."""

from prism.uncertainty.logprob import logprob_confidence, mean_logprob
from prism.uncertainty.risk import (
    DEFAULT_RISK_WEIGHTS,
    build_uncertainty_scores,
    compute_reliability_score,
    compute_risk_score,
    risk_level_from_score,
)
from prism.uncertainty.self_consistency import (
    average_pairwise_consistency,
    exact_match_consistency,
    simple_jaccard_similarity,
)
from prism.uncertainty.semantic_entropy import (
    normalized_semantic_entropy,
    semantic_entropy_from_clusters,
    semantic_entropy_from_samples,
)

__all__ = [
    "DEFAULT_RISK_WEIGHTS",
    "average_pairwise_consistency",
    "build_uncertainty_scores",
    "compute_reliability_score",
    "compute_risk_score",
    "exact_match_consistency",
    "logprob_confidence",
    "mean_logprob",
    "normalized_semantic_entropy",
    "risk_level_from_score",
    "semantic_entropy_from_clusters",
    "semantic_entropy_from_samples",
    "simple_jaccard_similarity",
]
