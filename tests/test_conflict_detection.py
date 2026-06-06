from datetime import datetime, timezone

from prism import ClaimStatus, ClaimTriple, ConflictSeverity, ConflictType
from prism.memory import (
    ClaimSupport,
    MemoryGraphStore,
    classify_claim_support,
    detect_duplicate_claim,
    find_claim_conflicts,
    normalize_claim,
    normalize_relation,
    normalize_text,
)


NOW = datetime(2026, 6, 6, 19, 0, tzinfo=timezone.utc)


def claim(
    subject: str,
    relation: str,
    object_: str,
    *,
    claim_id: str = "claim",
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


def test_normalization_handles_case_whitespace_punctuation_and_aliases() -> None:
    noisy = claim("  PRISM, Inc.  ", " Is Not ", "  Qwen-2.5!  ")

    assert normalize_text("  PRISM, Inc.  ") == "prism inc"
    assert normalize_relation(" Is Not ") == "is_not"
    assert normalize_relation("doesn't have") == "does_not_have"
    assert normalize_claim(noisy) == ("prism inc", "is_not", "qwen 2 5")


def test_duplicate_detection_uses_normalized_triple() -> None:
    existing = claim("PRISM", "is", "Prototype", claim_id="existing")
    new = claim(" prism! ", " same-as ", " prototype ", claim_id="new")

    assert detect_duplicate_claim(new, [existing]) is existing
    assert classify_claim_support(new, [existing]) is ClaimSupport.DUPLICATE


def test_direct_contradiction_is_and_is_not() -> None:
    existing = claim("PRISM", "is", "prototype", claim_id="existing")
    new = claim("PRISM", "is_not", "prototype", claim_id="new")

    assert classify_claim_support(new, [existing]) is ClaimSupport.CONTRADICTED
    conflicts = find_claim_conflicts(new, [existing])

    assert len(conflicts) == 1
    assert conflicts[0].conflict_type is ConflictType.CONTRADICTION
    assert conflicts[0].severity is ConflictSeverity.HIGH
    assert conflicts[0].evidence["rule"] == "direct_relation_contradiction"


def test_direct_contradiction_has_and_does_not_have() -> None:
    existing = claim("PRISM", "has", "memory graph", claim_id="existing")
    new = claim("PRISM", "does_not_have", "memory graph", claim_id="new")

    assert classify_claim_support(new, [existing]) is ClaimSupport.CONTRADICTED
    assert find_claim_conflicts(new, [existing])[0].severity is ConflictSeverity.HIGH


def test_functional_relation_conflict() -> None:
    existing = claim("PRISM", "default_model", "Qwen2.5", claim_id="existing")
    new = claim("PRISM", "default model", "Gemma", claim_id="new")

    assert classify_claim_support(new, [existing]) is ClaimSupport.CONTRADICTED
    conflicts = find_claim_conflicts(new, [existing])

    assert len(conflicts) == 1
    assert conflicts[0].severity is ConflictSeverity.MEDIUM
    assert conflicts[0].evidence["rule"] == "functional_relation_conflict"


def test_supported_accepted_claim() -> None:
    existing = claim(
        "PRISM",
        "is",
        "prototype",
        claim_id="existing",
        status=ClaimStatus.ACCEPTED,
    )
    new = claim("prism", "equals", "prototype", claim_id="new")

    assert classify_claim_support(new, [existing]) is ClaimSupport.SUPPORTED


def test_unknown_claim() -> None:
    existing = claim("Semantic Entropy", "is", "uncertainty metric", claim_id="existing")
    new = claim("PRISM", "uses", "claim triples", claim_id="new")

    assert classify_claim_support(new, [existing]) is ClaimSupport.UNKNOWN
    assert find_claim_conflicts(new, [existing]) == ()


def test_weakly_supported_related_entity_and_relation() -> None:
    existing = claim("PRISM", "uses", "knowledge graph memory", claim_id="existing")
    new = claim("PRISM", "uses", "claim triples", claim_id="new")

    assert classify_claim_support(new, [existing]) is ClaimSupport.WEAKLY_SUPPORTED
    assert find_claim_conflicts(new, [existing]) == ()


def test_memory_conflict_object_creation() -> None:
    existing = claim("PRISM", "supports", "KG memory", claim_id="existing")
    new = claim("PRISM", "contradicts", "KG memory", claim_id="new")

    conflict = find_claim_conflicts(new, [existing])[0]

    assert conflict.new_claim is new
    assert conflict.existing_claim is existing
    assert conflict.conflict_type is ConflictType.CONTRADICTION
    assert conflict.confidence == 1.0
    assert "Direct contradiction" in conflict.explanation


def test_no_false_contradiction_for_unrelated_claims() -> None:
    existing = claim("PRISM", "is", "prototype", claim_id="existing")
    new = claim("SelfCheckGPT", "is_not", "prototype", claim_id="new")

    assert classify_claim_support(new, [existing]) is ClaimSupport.UNKNOWN
    assert find_claim_conflicts(new, [existing]) == ()


def test_memory_graph_store_conflict_helpers() -> None:
    store = MemoryGraphStore(
        [
            claim("PRISM", "default_model", "Qwen2.5", claim_id="default-model"),
            claim("PRISM", "uses", "knowledge graph memory", claim_id="uses-kg"),
        ]
    )
    new = claim("PRISM", "default_model", "Gemma", claim_id="new-default")

    assert store.check_claim_support(new) is ClaimSupport.CONTRADICTED
    assert [related.claim_id for related in store.find_related_claims(new)] == ["default-model"]
    conflicts = store.find_conflicts(new)
    assert len(conflicts) == 1
    assert conflicts[0].existing_claim.claim_id == "default-model"
