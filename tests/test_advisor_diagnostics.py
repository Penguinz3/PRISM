from datetime import datetime, timezone

import pytest

from prism import (
    AdvisorDiagnostic,
    ClaimStatus,
    ClaimTriple,
    ConflictSeverity,
    ConflictType,
    MemoryConflict,
    RiskLevel,
    UncertaintyScores,
)
from prism.advisor import (
    ACTION_REJECT_OR_DO_NOT_STORE,
    ACTION_TRUST,
    ACTION_VERIFY_BEFORE_TRUSTING,
    LABEL_CONTRADICTED,
    LABEL_DUPLICATE,
    LABEL_NEEDS_VERIFICATION,
    LABEL_SAFE_TO_STORE,
    LABEL_UNKNOWN,
    build_advisor_diagnostic,
    build_revision_instruction,
    classify_risk_level,
    diagnose_claims,
    recommend_action,
    summarize_conflicts,
    summarize_uncertainty,
)
from prism.memory import ClaimSupport


NOW = datetime(2026, 6, 6, 20, 0, tzinfo=timezone.utc)


def claim(
    subject: str = "PRISM",
    relation: str = "is",
    object_: str = "prototype",
    *,
    claim_id: str = "claim-1",
    status: ClaimStatus | str = ClaimStatus.PROPOSED,
) -> ClaimTriple:
    return ClaimTriple(
        subject=subject,
        relation=relation,
        object=object_,
        confidence=0.9,
        source="unit-test",
        timestamp=NOW,
        status=status,
        claim_id=claim_id,
    )


def conflict_for(new_claim: ClaimTriple, existing_claim: ClaimTriple) -> MemoryConflict:
    return MemoryConflict(
        new_claim=new_claim,
        existing_claim=existing_claim,
        conflict_type=ConflictType.CONTRADICTION,
        severity=ConflictSeverity.HIGH,
        confidence=1.0,
        explanation="Direct contradiction for test.",
    )


def test_low_medium_high_risk_classification() -> None:
    assert classify_risk_level(0.1) is RiskLevel.LOW
    assert classify_risk_level(0.33) is RiskLevel.MEDIUM
    assert classify_risk_level(0.66) is RiskLevel.HIGH
    assert classify_risk_level(None) is RiskLevel.UNKNOWN


def test_trust_recommendation_with_no_conflicts() -> None:
    assert recommend_action(RiskLevel.LOW, conflicts=[], unsupported_claims=[]) == ACTION_TRUST


def test_trust_recommendation_with_unsupported_claims() -> None:
    unsupported = [claim(claim_id="unsupported")]

    assert (
        recommend_action(RiskLevel.LOW, conflicts=[], unsupported_claims=unsupported)
        == ACTION_VERIFY_BEFORE_TRUSTING
    )


def test_trust_recommendation_with_contradictions() -> None:
    new = claim(claim_id="new")
    existing = claim(object_="different", claim_id="existing")
    conflict = conflict_for(new, existing)

    assert (
        recommend_action(RiskLevel.LOW, conflicts=[conflict], unsupported_claims=[])
        == ACTION_REJECT_OR_DO_NOT_STORE
    )


def test_claim_diagnostic_labels() -> None:
    safe = claim(claim_id="safe")
    weak = claim(claim_id="weak")
    contradicted = claim(claim_id="contradicted")
    duplicate = claim(claim_id="duplicate")
    unknown = claim(claim_id="unknown")
    existing = claim(object_="existing", claim_id="existing")
    conflict = conflict_for(contradicted, existing)

    diagnostics = diagnose_claims(
        [safe, weak, contradicted, duplicate, unknown],
        {
            "safe": ClaimSupport.SUPPORTED,
            "weak": ClaimSupport.WEAKLY_SUPPORTED,
            "contradicted": ClaimSupport.SUPPORTED,
            "duplicate": ClaimSupport.DUPLICATE,
            "unknown": ClaimSupport.UNKNOWN,
        },
        [conflict],
    )

    labels = {diagnostic["claim_id"]: diagnostic["label"] for diagnostic in diagnostics}
    assert labels == {
        "safe": LABEL_SAFE_TO_STORE,
        "weak": LABEL_NEEDS_VERIFICATION,
        "contradicted": LABEL_CONTRADICTED,
        "duplicate": LABEL_DUPLICATE,
        "unknown": LABEL_UNKNOWN,
    }


def test_conflict_summary_creation() -> None:
    new = claim(claim_id="new")
    existing = claim(object_="existing", claim_id="existing")
    conflict = conflict_for(new, existing)

    summaries = summarize_conflicts([conflict])

    assert summaries[0]["conflict_type"] == "contradiction"
    assert summaries[0]["severity"] == "high"
    assert summaries[0]["new_claim"]["claim_id"] == "new"
    assert summaries[0]["existing_claim"]["claim_id"] == "existing"
    assert summaries[0]["explanation"] == "Direct contradiction for test."


def test_uncertainty_summary_creation() -> None:
    scores = UncertaintyScores(
        mean_token_probability=0.4,
        self_consistency_disagreement=0.75,
        semantic_entropy=1.2,
        semantic_entropy_normalized=0.8,
        kg_conflict_score=0.2,
        unsupported_claim_score=0.1,
        combined_risk_score=0.7,
        details={"kg_support": 0.3},
    )

    summary = summarize_uncertainty(scores)

    assert summary["logprob_confidence"] == 0.4
    assert summary["self_consistency"] == 0.25
    assert summary["kg_support"] == 0.3
    assert summary["reliability"] == pytest.approx(0.3)
    assert summary["risk"] == 0.7
    assert "high_risk_score" in summary["strongest_warning_signals"]
    assert "self_consistency_disagreement" in summary["strongest_warning_signals"]


def test_advisor_diagnostic_build() -> None:
    safe = claim(claim_id="safe")
    risky = claim(claim_id="risky")
    scores = UncertaintyScores(
        self_consistency_disagreement=0.2,
        combined_risk_score=0.4,
        details={"kg_support": 0.5},
    )

    diagnostic = build_advisor_diagnostic(
        prompt="What is PRISM?",
        answer="PRISM is a prototype.",
        claims=[safe, risky],
        uncertainty_scores=scores,
        conflicts=[],
        support_classifications={
            "safe": ClaimSupport.SUPPORTED,
            "risky": ClaimSupport.UNKNOWN,
        },
    )

    assert isinstance(diagnostic, AdvisorDiagnostic)
    assert diagnostic.risk_level is RiskLevel.MEDIUM
    assert diagnostic.metadata["recommended_action"] == ACTION_VERIFY_BEFORE_TRUSTING
    assert diagnostic.claims_safe_to_store == (safe,)
    assert diagnostic.claims_requiring_verification == (risky,)
    assert diagnostic.metadata["memory_update_recommendation"]["safe_to_store_claim_ids"] == ["safe"]
    assert diagnostic.metadata["memory_update_recommendation"]["requires_verification_claim_ids"] == [
        "risky"
    ]


def test_revision_instruction_includes_contradicted_and_unsupported_guidance() -> None:
    diagnostic_metadata = {
        "recommended_action": ACTION_REJECT_OR_DO_NOT_STORE,
        "claim_diagnostics": [
            {"claim_id": "a", "label": LABEL_CONTRADICTED},
            {"claim_id": "b", "label": LABEL_UNKNOWN},
        ],
    }

    instruction = build_revision_instruction(diagnostic_metadata)

    assert "avoid contradicted claims" in instruction
    assert "unsupported or weakly supported claims as uncertain" in instruction
    assert "Do not store claims requiring verification" in instruction


def test_memory_update_recommendation_blocks_contradicted_claims() -> None:
    contradicted = claim(claim_id="contradicted")
    existing = claim(object_="existing", claim_id="existing")
    conflict = conflict_for(contradicted, existing)
    scores = UncertaintyScores(combined_risk_score=0.8)

    diagnostic = build_advisor_diagnostic(
        prompt="What is PRISM?",
        answer="PRISM is complete.",
        claims=[contradicted],
        uncertainty_scores=scores,
        conflicts=[conflict],
        support_classifications={"contradicted": ClaimSupport.CONTRADICTED},
    )

    recommendation = diagnostic.metadata["memory_update_recommendation"]
    assert recommendation["block_contradicted_claims"] is True
    assert recommendation["blocked_claim_ids"] == ["contradicted"]
    assert recommendation["action"] == "do_not_store_until_review"
