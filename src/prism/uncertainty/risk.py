"""Reliability and risk aggregation helpers."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from prism.schemas import RiskLevel, TokenConfidence, UncertaintyScores
from prism.uncertainty.logprob import logprob_confidence as logprob_confidence_score
from prism.uncertainty.logprob import mean_logprob
from prism.uncertainty.self_consistency import average_pairwise_consistency
from prism.uncertainty.semantic_entropy import (
    normalized_semantic_entropy as normalized_entropy_score,
)
from prism.uncertainty.semantic_entropy import semantic_entropy_from_clusters


DEFAULT_RISK_WEIGHTS = {
    "logprob": 1.0,
    "self_consistency": 1.0,
    "kg_support": 1.0,
    "entropy": 1.0,
}


def _clamp_probability(value: float) -> float:
    return max(0.0, min(1.0, value))


def _validate_probability(name: str, value: float | None) -> float | None:
    if value is None:
        return None
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0.0 and 1.0")
    return value


def _weights(weights: Mapping[str, float] | None) -> dict[str, float]:
    merged = dict(DEFAULT_RISK_WEIGHTS)
    if weights:
        merged.update(weights)
    for key, value in merged.items():
        if value < 0:
            raise ValueError(f"weight {key} must be non-negative")
    return merged


def compute_reliability_score(
    *,
    logprob_confidence: float | None = None,
    self_consistency: float | None = None,
    kg_support: float | None = None,
    normalized_semantic_entropy: float | None = None,
    weights: Mapping[str, float] | None = None,
) -> float | None:
    """Compute normalized reliability from available PRISM signals.

    The raw formula follows the formal definition:

    `Rel = w_l * logprob + w_s * self_consistency + w_k * kg_support - w_h * entropy`.

    The raw score is normalized over the theoretical range of the signals that
    are actually provided, then clamped to `[0, 1]`.
    """

    logprob_confidence = _validate_probability("logprob_confidence", logprob_confidence)
    self_consistency = _validate_probability("self_consistency", self_consistency)
    kg_support = _validate_probability("kg_support", kg_support)
    normalized_semantic_entropy = _validate_probability(
        "normalized_semantic_entropy",
        normalized_semantic_entropy,
    )
    active_weights = _weights(weights)

    raw = 0.0
    min_raw = 0.0
    max_raw = 0.0
    has_signal = False

    positive_signals = (
        ("logprob", logprob_confidence),
        ("self_consistency", self_consistency),
        ("kg_support", kg_support),
    )
    for key, value in positive_signals:
        if value is None:
            continue
        weight = active_weights[key]
        raw += weight * value
        max_raw += weight
        has_signal = True

    if normalized_semantic_entropy is not None:
        weight = active_weights["entropy"]
        raw -= weight * normalized_semantic_entropy
        min_raw -= weight
        has_signal = True

    if not has_signal:
        return None
    if max_raw == min_raw:
        return 0.0

    normalized = (raw - min_raw) / (max_raw - min_raw)
    return _clamp_probability(normalized)


def compute_risk_score(
    *,
    logprob_confidence: float | None = None,
    self_consistency: float | None = None,
    kg_support: float | None = None,
    normalized_semantic_entropy: float | None = None,
    weights: Mapping[str, float] | None = None,
) -> float | None:
    reliability = compute_reliability_score(
        logprob_confidence=logprob_confidence,
        self_consistency=self_consistency,
        kg_support=kg_support,
        normalized_semantic_entropy=normalized_semantic_entropy,
        weights=weights,
    )
    if reliability is None:
        return None
    return _clamp_probability(1.0 - reliability)


def risk_level_from_score(score: float | None) -> RiskLevel:
    if score is None:
        return RiskLevel.UNKNOWN
    score = _clamp_probability(score)
    if score < 0.25:
        return RiskLevel.LOW
    if score < 0.5:
        return RiskLevel.MEDIUM
    if score < 0.75:
        return RiskLevel.HIGH
    return RiskLevel.CRITICAL


def build_uncertainty_scores(
    *,
    token_confidences: Iterable[TokenConfidence | Mapping[str, Any]] | None = None,
    samples=None,
    cluster_sizes: Iterable[int] | None = None,
    kg_support: float | None = None,
    weights: Mapping[str, float] | None = None,
    details: Mapping[str, Any] | None = None,
) -> UncertaintyScores:
    """Build an `UncertaintyScores` object from available lightweight signals."""

    token_confidences = tuple(token_confidences or ())
    mean_logprob_value = mean_logprob(token_confidences)
    logprob_confidence_value = logprob_confidence_score(token_confidences)

    consistency = average_pairwise_consistency(samples) if samples is not None else None
    disagreement = None if consistency is None else _clamp_probability(1.0 - consistency)

    entropy = None
    normalized_entropy = None
    if cluster_sizes is not None:
        cluster_sizes = tuple(cluster_sizes)
        entropy = semantic_entropy_from_clusters(cluster_sizes)
        normalized_entropy = normalized_entropy_score(cluster_sizes)

    risk = compute_risk_score(
        logprob_confidence=logprob_confidence_value,
        self_consistency=consistency,
        kg_support=kg_support,
        normalized_semantic_entropy=normalized_entropy,
        weights=weights,
    )

    score_details = dict(details or {})
    if kg_support is not None:
        score_details["kg_support"] = kg_support
    if consistency is not None:
        score_details["self_consistency"] = consistency
    if weights is not None:
        score_details["weights"] = dict(weights)

    return UncertaintyScores(
        mean_logprob=mean_logprob_value,
        mean_token_probability=logprob_confidence_value,
        self_consistency_disagreement=disagreement,
        semantic_entropy=entropy,
        semantic_entropy_normalized=normalized_entropy,
        combined_risk_score=risk,
        risk_level=risk_level_from_score(risk),
        details=score_details,
    )
