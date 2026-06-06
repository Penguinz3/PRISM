"""Deterministic advisor diagnostics for PRISM.

This layer prepares structured diagnostics for a future second-pass advisor. It
does not call an LLM and does not mutate KG memory.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from prism.memory import ClaimSupport
from prism.schemas import (
    AdvisorDiagnostic,
    ClaimTriple,
    ConflictType,
    MemoryConflict,
    RiskLevel,
    UncertaintyScores,
)


ACTION_TRUST = "trust"
ACTION_VERIFY_BEFORE_TRUSTING = "verify_before_trusting"
ACTION_REVISE_BEFORE_TRUSTING = "revise_before_trusting"
ACTION_REJECT_OR_DO_NOT_STORE = "reject_or_do_not_store"

LABEL_SAFE_TO_STORE = "safe_to_store"
LABEL_NEEDS_VERIFICATION = "needs_verification"
LABEL_CONTRADICTED = "contradicted"
LABEL_DUPLICATE = "duplicate"
LABEL_UNKNOWN = "unknown"


def _clamp_probability(value: float) -> float:
    return max(0.0, min(1.0, value))


def _risk_value(risk_score: float | None, reliability_score: float | None = None) -> float | None:
    if risk_score is not None:
        return _clamp_probability(risk_score)
    if reliability_score is not None:
        return _clamp_probability(1.0 - reliability_score)
    return None


def _risk_level_value(risk_level: RiskLevel | str) -> str:
    if isinstance(risk_level, RiskLevel):
        return risk_level.value
    return str(risk_level)


def _count_items(value: int | Sequence[Any] | None) -> int:
    if value is None:
        return 0
    if isinstance(value, int):
        return value
    return len(value)


def classify_risk_level(
    risk_score: float | None,
    *,
    thresholds: tuple[float, float] = (0.33, 0.66),
) -> RiskLevel:
    """Classify risk into low, medium, high, or unknown."""

    if risk_score is None:
        return RiskLevel.UNKNOWN
    low_threshold, high_threshold = thresholds
    if not 0.0 <= low_threshold < high_threshold <= 1.0:
        raise ValueError("thresholds must satisfy 0.0 <= low < high <= 1.0")
    score = _clamp_probability(risk_score)
    if score < low_threshold:
        return RiskLevel.LOW
    if score < high_threshold:
        return RiskLevel.MEDIUM
    return RiskLevel.HIGH


def recommend_action(
    risk_level: RiskLevel | str,
    conflicts: Sequence[MemoryConflict] | None = None,
    unsupported_claims: int | Sequence[ClaimTriple] | None = None,
) -> str:
    """Recommend a trust action without mutating memory."""

    conflicts = tuple(conflicts or ())
    unsupported_count = _count_items(unsupported_claims)
    has_contradiction = any(
        conflict.conflict_type is ConflictType.CONTRADICTION for conflict in conflicts
    )
    normalized_risk = _risk_level_value(risk_level)

    if has_contradiction:
        return ACTION_REJECT_OR_DO_NOT_STORE
    if normalized_risk in {RiskLevel.HIGH.value, RiskLevel.CRITICAL.value}:
        return ACTION_REVISE_BEFORE_TRUSTING
    if normalized_risk == RiskLevel.MEDIUM.value or unsupported_count > 0:
        return ACTION_VERIFY_BEFORE_TRUSTING
    return ACTION_TRUST


def _coerce_support(value: ClaimSupport | str | None) -> ClaimSupport:
    if value is None:
        return ClaimSupport.UNKNOWN
    if isinstance(value, ClaimSupport):
        return value
    return ClaimSupport(value)


def _support_for_claim(
    claim: ClaimTriple,
    index: int,
    support_results: Mapping[str, ClaimSupport | str] | Sequence[ClaimSupport | str] | None,
) -> ClaimSupport:
    if support_results is None:
        return ClaimSupport.UNKNOWN
    if isinstance(support_results, Mapping):
        return _coerce_support(support_results.get(claim.claim_id))
    return _coerce_support(support_results[index] if index < len(support_results) else None)


def _conflicted_claim_ids(conflicts: Sequence[MemoryConflict]) -> set[str]:
    return {conflict.new_claim.claim_id for conflict in conflicts}


def diagnose_claims(
    claims: Sequence[ClaimTriple],
    support_results: Mapping[str, ClaimSupport | str] | Sequence[ClaimSupport | str] | None = None,
    conflicts: Sequence[MemoryConflict] | None = None,
) -> list[dict[str, Any]]:
    """Label claims for storage or verification recommendations."""

    conflicts = tuple(conflicts or ())
    contradicted_ids = _conflicted_claim_ids(conflicts)
    diagnostics: list[dict[str, Any]] = []

    for index, claim in enumerate(claims):
        support = _support_for_claim(claim, index, support_results)
        reasons: list[str] = []

        if claim.claim_id in contradicted_ids or support is ClaimSupport.CONTRADICTED:
            label = LABEL_CONTRADICTED
            reasons.append("claim contradicts KG memory")
        elif support is ClaimSupport.DUPLICATE:
            label = LABEL_DUPLICATE
            reasons.append("claim already exists in memory")
        elif support is ClaimSupport.SUPPORTED:
            label = LABEL_SAFE_TO_STORE
            reasons.append("claim is supported by memory")
        elif support is ClaimSupport.WEAKLY_SUPPORTED:
            label = LABEL_NEEDS_VERIFICATION
            reasons.append("claim has weak memory support")
        else:
            label = LABEL_UNKNOWN
            reasons.append("claim support is unknown")

        diagnostics.append(
            {
                "claim_id": claim.claim_id,
                "claim": claim.to_dict(),
                "label": label,
                "support": support.value,
                "reasons": reasons,
            }
        )

    return diagnostics


def summarize_conflicts(conflicts: Sequence[MemoryConflict] | None) -> list[dict[str, Any]]:
    """Return concise structured summaries for KG conflicts."""

    return [
        {
            "conflict_id": conflict.conflict_id,
            "conflict_type": conflict.conflict_type.value,
            "severity": conflict.severity.value,
            "new_claim": conflict.new_claim.to_dict(),
            "existing_claim": conflict.existing_claim.to_dict(),
            "explanation": conflict.explanation,
        }
        for conflict in (conflicts or ())
    ]


def summarize_uncertainty(scores: UncertaintyScores) -> dict[str, Any]:
    """Summarize uncertainty fields and strongest warning signals."""

    self_consistency = None
    if scores.self_consistency_disagreement is not None:
        self_consistency = _clamp_probability(1.0 - scores.self_consistency_disagreement)

    risk = scores.combined_risk_score
    reliability = None if risk is None else _clamp_probability(1.0 - risk)
    kg_support = scores.details.get("kg_support")

    warning_signals: list[str] = []
    if risk is not None and risk >= 0.66:
        warning_signals.append("high_risk_score")
    if scores.mean_token_probability is not None and scores.mean_token_probability < 0.5:
        warning_signals.append("low_logprob_confidence")
    if (
        scores.self_consistency_disagreement is not None
        and scores.self_consistency_disagreement >= 0.5
    ):
        warning_signals.append("self_consistency_disagreement")
    if (
        scores.semantic_entropy_normalized is not None
        and scores.semantic_entropy_normalized >= 0.5
    ):
        warning_signals.append("semantic_entropy")
    if scores.kg_conflict_score is not None and scores.kg_conflict_score > 0:
        warning_signals.append("kg_conflict_score")
    if scores.unsupported_claim_score is not None and scores.unsupported_claim_score > 0:
        warning_signals.append("unsupported_claim_score")

    return {
        "logprob_confidence": scores.mean_token_probability,
        "self_consistency": self_consistency,
        "semantic_entropy": scores.semantic_entropy,
        "normalized_semantic_entropy": scores.semantic_entropy_normalized,
        "kg_support": kg_support,
        "reliability": reliability,
        "risk": risk,
        "strongest_warning_signals": warning_signals,
    }


def _claims_with_label(
    claims: Sequence[ClaimTriple],
    claim_diagnostics: Sequence[Mapping[str, Any]],
    labels: set[str],
) -> list[ClaimTriple]:
    by_id = {claim.claim_id: claim for claim in claims}
    return [
        by_id[diagnostic["claim_id"]]
        for diagnostic in claim_diagnostics
        if diagnostic["label"] in labels and diagnostic["claim_id"] in by_id
    ]


def _memory_update_recommendation(
    recommended_action: str,
    claim_diagnostics: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    contradicted_ids = [
        diagnostic["claim_id"]
        for diagnostic in claim_diagnostics
        if diagnostic["label"] == LABEL_CONTRADICTED
    ]
    safe_ids = [
        diagnostic["claim_id"]
        for diagnostic in claim_diagnostics
        if diagnostic["label"] == LABEL_SAFE_TO_STORE
    ]
    verification_ids = [
        diagnostic["claim_id"]
        for diagnostic in claim_diagnostics
        if diagnostic["label"] in {LABEL_NEEDS_VERIFICATION, LABEL_UNKNOWN}
    ]
    duplicate_ids = [
        diagnostic["claim_id"]
        for diagnostic in claim_diagnostics
        if diagnostic["label"] == LABEL_DUPLICATE
    ]

    return {
        "action": (
            "store_safe_claims_only"
            if recommended_action == ACTION_TRUST
            else "do_not_store_until_review"
        ),
        "safe_to_store_claim_ids": safe_ids,
        "requires_verification_claim_ids": verification_ids,
        "duplicate_claim_ids": duplicate_ids,
        "blocked_claim_ids": contradicted_ids,
        "block_contradicted_claims": bool(contradicted_ids),
    }


def build_revision_instruction(diagnostic: AdvisorDiagnostic | Mapping[str, Any]) -> str:
    """Build plain text instructions for a future second-pass model."""

    if isinstance(diagnostic, AdvisorDiagnostic):
        metadata = diagnostic.metadata
        action = metadata.get("recommended_action", ACTION_VERIFY_BEFORE_TRUSTING)
        claim_diagnostics = metadata.get("claim_diagnostics", ())
    else:
        metadata = diagnostic
        action = metadata.get("recommended_action", ACTION_VERIFY_BEFORE_TRUSTING)
        claim_diagnostics = metadata.get("claim_diagnostics", ())

    labels = {item.get("label") for item in claim_diagnostics}
    parts = [f"Recommended action: {action}."]
    if LABEL_CONTRADICTED in labels:
        parts.append("Revise the answer to avoid contradicted claims.")
    if LABEL_NEEDS_VERIFICATION in labels or LABEL_UNKNOWN in labels:
        parts.append("Mark unsupported or weakly supported claims as uncertain.")
    if LABEL_DUPLICATE in labels:
        parts.append("Do not store duplicate claims again.")
    parts.append("Do not store claims requiring verification.")
    return " ".join(parts)


def build_advisor_diagnostic(
    *,
    prompt: str,
    answer: str,
    claims: Sequence[ClaimTriple],
    uncertainty_scores: UncertaintyScores,
    risk_score: float | None = None,
    reliability_score: float | None = None,
    conflicts: Sequence[MemoryConflict] | None = None,
    support_classifications: (
        Mapping[str, ClaimSupport | str] | Sequence[ClaimSupport | str] | None
    ) = None,
    risk_thresholds: tuple[float, float] = (0.33, 0.66),
) -> AdvisorDiagnostic:
    """Build a deterministic advisor diagnostic packet."""

    conflicts = tuple(conflicts or ())
    risk = _risk_value(risk_score, reliability_score)
    if risk is None:
        risk = uncertainty_scores.combined_risk_score
    risk_level = classify_risk_level(risk, thresholds=risk_thresholds)

    claim_diagnostics = diagnose_claims(claims, support_classifications, conflicts)
    risky_labels = {LABEL_CONTRADICTED, LABEL_NEEDS_VERIFICATION, LABEL_UNKNOWN}
    risky_claims = _claims_with_label(claims, claim_diagnostics, risky_labels)
    safe_claims = _claims_with_label(claims, claim_diagnostics, {LABEL_SAFE_TO_STORE})
    unsupported_claims = _claims_with_label(
        claims,
        claim_diagnostics,
        {LABEL_NEEDS_VERIFICATION, LABEL_UNKNOWN},
    )

    action = recommend_action(risk_level, conflicts, unsupported_claims)
    uncertainty_summary = summarize_uncertainty(uncertainty_scores)
    conflict_summaries = summarize_conflicts(conflicts)
    memory_update = _memory_update_recommendation(action, claim_diagnostics)

    metadata = {
        "prompt": prompt,
        "answer": answer,
        "recommended_action": action,
        "risky_claims": [claim.to_dict() for claim in risky_claims],
        "safe_claims": [claim.to_dict() for claim in safe_claims],
        "claim_diagnostics": claim_diagnostics,
        "conflict_summaries": conflict_summaries,
        "uncertainty_summary": uncertainty_summary,
        "memory_update_recommendation": memory_update,
    }
    revision_instruction = build_revision_instruction(metadata)
    metadata["revision_instruction"] = revision_instruction

    summary = (
        f"Risk level {risk_level.value}; recommended action {action}; "
        f"{len(safe_claims)} safe claim(s), {len(risky_claims)} risky claim(s), "
        f"{len(conflicts)} conflict(s)."
    )

    return AdvisorDiagnostic(
        risk_level=risk_level,
        summary=summary,
        risk_score=risk,
        low_confidence_markers=uncertainty_summary["strongest_warning_signals"],
        semantic_disagreement_summary=(
            None
            if uncertainty_summary["self_consistency"] is None
            else f"Self-consistency: {uncertainty_summary['self_consistency']:.3f}"
        ),
        kg_conflicts=conflicts,
        claims_safe_to_store=safe_claims,
        claims_requiring_verification=risky_claims,
        suggested_revised_answer_instruction=revision_instruction,
        metadata=metadata,
    )
