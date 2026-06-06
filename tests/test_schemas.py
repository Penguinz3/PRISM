import json
from datetime import datetime, timezone

import pytest

from prism import (
    AdvisorDiagnostic,
    ClaimStatus,
    ClaimTriple,
    ConflictSeverity,
    ConflictType,
    GeneratedAnswer,
    MemoryConflict,
    PRISMRunResult,
    RiskLevel,
    SampleSet,
    TokenConfidence,
    UncertaintyScores,
)


NOW = datetime(2026, 6, 6, 17, 30, tzinfo=timezone.utc)


def test_generated_answer_keeps_token_confidence_and_round_trips() -> None:
    token = TokenConfidence(token="Paris", logprob=-0.2, probability=0.82, rank=1)
    answer = GeneratedAnswer(
        prompt="Where is the Eiffel Tower?",
        text="The Eiffel Tower is in Paris.",
        answer_id="answer-1",
        model_name="mock-model",
        created_at=NOW,
        token_confidences=[token],
        sampling_parameters={"temperature": 0.7},
    )

    payload = answer.to_dict()
    json.dumps(payload)
    restored = GeneratedAnswer.from_dict(payload)

    assert restored.answer_id == "answer-1"
    assert restored.token_confidences[0].probability == 0.82
    assert restored.sampling_parameters["temperature"] == 0.7


def test_generated_answer_rejects_empty_text() -> None:
    with pytest.raises(ValueError, match="text"):
        GeneratedAnswer(prompt="Question?", text=" ")


def test_sample_set_counts_samples_and_includes_primary() -> None:
    primary = GeneratedAnswer(prompt="Q", text="A", answer_id="primary")
    sample = GeneratedAnswer(prompt="Q", text="B", answer_id="sample")
    sample_set = SampleSet(prompt="Q", primary_answer=primary, samples=[sample])

    assert sample_set.sample_count == 1
    assert [answer.answer_id for answer in sample_set.all_answers] == ["primary", "sample"]


def test_uncertainty_scores_validate_ranges_and_expose_risk_vector() -> None:
    scores = UncertaintyScores(
        mean_logprob=-0.4,
        self_consistency_disagreement=0.25,
        semantic_entropy=1.2,
        kg_conflict_score=0.5,
        unsupported_claim_score=0.1,
        combined_risk_score=0.42,
        risk_level=RiskLevel.MEDIUM,
    )

    assert scores.risk_vector == {"L": -0.4, "S": 0.25, "H": 1.2, "K": 0.5, "U": 0.1}

    with pytest.raises(ValueError, match="self_consistency_disagreement"):
        UncertaintyScores(self_consistency_disagreement=1.1)

    with pytest.raises(ValueError, match="semantic_entropy"):
        UncertaintyScores(semantic_entropy=-0.1)


def test_claim_triple_validates_and_normalizes_canonical_key() -> None:
    claim = ClaimTriple(
        subject="  Eiffel   Tower ",
        relation=" Located_In ",
        object=" Paris ",
        confidence=0.9,
        source="unit-test",
        timestamp=NOW,
        status=ClaimStatus.ACCEPTED,
        claim_id="claim-1",
    )

    assert claim.subject == "Eiffel   Tower"
    assert claim.canonical_key == ("eiffel tower", "located_in", "paris")
    assert claim.to_dict()["status"] == "accepted"

    with pytest.raises(ValueError, match="confidence"):
        ClaimTriple(subject="A", relation="is", object="B", confidence=1.2)


def test_memory_conflict_serializes_nested_claims() -> None:
    new_claim = ClaimTriple(subject="PRISM", relation="status", object="complete", claim_id="new")
    existing_claim = ClaimTriple(
        subject="PRISM",
        relation="status",
        object="prototype",
        claim_id="existing",
    )
    conflict = MemoryConflict(
        new_claim=new_claim,
        existing_claim=existing_claim,
        conflict_type=ConflictType.CONTRADICTION,
        severity=ConflictSeverity.HIGH,
        confidence=0.77,
        explanation="Same subject and relation, incompatible objects.",
        conflict_id="conflict-1",
        detected_at=NOW,
    )

    payload = conflict.to_dict()
    restored = MemoryConflict.from_dict(payload)

    assert restored.conflict_type is ConflictType.CONTRADICTION
    assert restored.severity is ConflictSeverity.HIGH
    assert restored.new_claim.object == "complete"
    assert restored.existing_claim.object == "prototype"


def test_advisor_diagnostic_groups_conflicts_and_claims() -> None:
    safe_claim = ClaimTriple(subject="PRISM", relation="is", object="prototype")
    risky_claim = ClaimTriple(subject="PRISM", relation="is", object="production")
    conflict = MemoryConflict(
        new_claim=risky_claim,
        existing_claim=safe_claim,
        conflict_type="contradiction",
    )
    diagnostic = AdvisorDiagnostic(
        risk_level="high",
        risk_score=0.8,
        summary="KG conflict requires verification.",
        low_confidence_markers=["semantic_entropy"],
        semantic_disagreement_summary="Samples split into multiple meanings.",
        kg_conflicts=[conflict],
        claims_safe_to_store=[safe_claim],
        claims_requiring_verification=[risky_claim],
        suggested_revised_answer_instruction="Answer with the verified prototype status only.",
    )

    assert diagnostic.risk_level is RiskLevel.HIGH
    assert diagnostic.kg_conflicts[0].conflict_type is ConflictType.CONTRADICTION
    assert diagnostic.claims_safe_to_store[0].object == "prototype"
    assert diagnostic.claims_requiring_verification[0].object == "production"


def test_prism_run_result_is_json_compatible_and_round_trips() -> None:
    primary = GeneratedAnswer(prompt="What is PRISM?", text="PRISM is a prototype.", answer_id="a1")
    sample = GeneratedAnswer(prompt="What is PRISM?", text="PRISM is a research system.", answer_id="s1")
    sample_set = SampleSet(prompt="What is PRISM?", primary_answer=primary, samples=[sample])
    claim = ClaimTriple(subject="PRISM", relation="is", object="prototype", claim_id="c1")
    scores = UncertaintyScores(combined_risk_score=0.2, risk_level=RiskLevel.LOW)
    diagnostic = AdvisorDiagnostic(risk_level=RiskLevel.LOW, summary="Low risk.", risk_score=0.2)
    result = PRISMRunResult(
        prompt="What is PRISM?",
        primary_answer=primary,
        run_id="run-1",
        sample_set=sample_set,
        uncertainty_scores=scores,
        extracted_claims=[claim],
        advisor_diagnostic=diagnostic,
        model_info={"name": "mock-model"},
        config={"phase": 1},
        timestamp=NOW,
    )

    payload = result.to_dict()
    json.dumps(payload)
    restored = PRISMRunResult.from_dict(payload)

    assert restored.run_id == "run-1"
    assert restored.sample_set is not None
    assert restored.sample_set.sample_count == 1
    assert restored.extracted_claims[0].canonical_key == ("prism", "is", "prototype")
    assert restored.uncertainty_scores.risk_level is RiskLevel.LOW
