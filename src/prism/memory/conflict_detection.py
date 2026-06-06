"""Deterministic KG support and conflict rules for PRISM.

This module is deliberately rule-based. It does not use NLI, embeddings, model
inference, semantic entropy, or fuzzy entity resolution.
"""

from __future__ import annotations

import string
from enum import Enum
from typing import Iterable

from prism.schemas import ClaimStatus, ClaimTriple, ConflictSeverity, ConflictType, MemoryConflict


class ClaimSupport(str, Enum):
    """Relationship between a new claim and existing graph memory."""

    DUPLICATE = "duplicate"
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    WEAKLY_SUPPORTED = "weakly_supported"
    UNKNOWN = "unknown"


FUNCTIONAL_RELATIONS = frozenset(
    {
        "default_model",
        "birth_date",
        "founded_date",
        "email",
        "phone",
        "current_status",
    }
)

RELATION_ALIASES = {
    "is": "is",
    "is_a": "is",
    "same_as": "is",
    "equals": "is",
    "equal_to": "is",
    "is_not": "is_not",
    "isnt": "is_not",
    "isnt_a": "is_not",
    "not": "is_not",
    "not_equal_to": "is_not",
    "has": "has",
    "contains": "has",
    "does_not_have": "does_not_have",
    "doesnt_have": "does_not_have",
    "lacks": "does_not_have",
    "supports": "supports",
    "support": "supports",
    "contradicts": "contradicts",
    "contradict": "contradicts",
    "true": "true",
    "false": "false",
    "default_model": "default_model",
    "birth_date": "birth_date",
    "date_of_birth": "birth_date",
    "founded_date": "founded_date",
    "email": "email",
    "email_address": "email",
    "phone": "phone",
    "phone_number": "phone",
    "current_status": "current_status",
    "status": "current_status",
}

DIRECT_CONTRADICTION_RELATIONS = frozenset(
    {
        frozenset(("is", "is_not")),
        frozenset(("has", "does_not_have")),
        frozenset(("supports", "contradicts")),
        frozenset(("true", "false")),
    }
)

_PUNCTUATION_TO_SPACE = str.maketrans(
    {char: " " for char in string.punctuation if char not in {"_"}}
)


def normalize_text(value: str) -> str:
    """Normalize entity/object text for deterministic comparison."""

    if not isinstance(value, str):
        raise TypeError("value must be a string")
    normalized = value.lower().strip()
    normalized = normalized.replace("'", "").replace("’", "").replace("`", "")
    normalized = normalized.translate(_PUNCTUATION_TO_SPACE)
    normalized = " ".join(normalized.split())
    if not normalized:
        raise ValueError("value must not be empty after normalization")
    return normalized


def normalize_relation(value: str) -> str:
    """Normalize relation text and apply a small deterministic alias table."""

    normalized = normalize_text(value).replace(" ", "_")
    return RELATION_ALIASES.get(normalized, normalized)


def normalize_claim(claim: ClaimTriple) -> tuple[str, str, str]:
    return (
        normalize_text(claim.subject),
        normalize_relation(claim.relation),
        normalize_text(claim.object),
    )


def detect_duplicate_claim(
    new_claim: ClaimTriple,
    existing_claims: Iterable[ClaimTriple],
) -> ClaimTriple | None:
    """Return the first existing claim with the same normalized triple."""

    new_key = normalize_claim(new_claim)
    for existing_claim in existing_claims:
        if normalize_claim(existing_claim) == new_key:
            return existing_claim
    return None


def relations_directly_contradict(relation_a: str, relation_b: str) -> bool:
    pair = frozenset((normalize_relation(relation_a), normalize_relation(relation_b)))
    return pair in DIRECT_CONTRADICTION_RELATIONS


def claims_directly_contradict(new_claim: ClaimTriple, existing_claim: ClaimTriple) -> bool:
    new_subject, new_relation, new_object = normalize_claim(new_claim)
    existing_subject, existing_relation, existing_object = normalize_claim(existing_claim)
    return (
        new_subject == existing_subject
        and new_object == existing_object
        and frozenset((new_relation, existing_relation)) in DIRECT_CONTRADICTION_RELATIONS
    )


def claims_have_functional_conflict(new_claim: ClaimTriple, existing_claim: ClaimTriple) -> bool:
    new_subject, new_relation, new_object = normalize_claim(new_claim)
    existing_subject, existing_relation, existing_object = normalize_claim(existing_claim)
    return (
        new_subject == existing_subject
        and new_relation == existing_relation
        and new_relation in FUNCTIONAL_RELATIONS
        and new_object != existing_object
    )


def claims_are_weakly_related(new_claim: ClaimTriple, existing_claim: ClaimTriple) -> bool:
    new_subject, new_relation, _new_object = normalize_claim(new_claim)
    existing_subject, existing_relation, _existing_object = normalize_claim(existing_claim)
    return (
        new_subject == existing_subject
        and (
            new_relation == existing_relation
            or frozenset((new_relation, existing_relation)) in DIRECT_CONTRADICTION_RELATIONS
        )
    )


def classify_claim_support(
    new_claim: ClaimTriple,
    existing_claims: Iterable[ClaimTriple],
) -> ClaimSupport:
    """Classify the strongest deterministic relationship to existing memory."""

    claims = tuple(existing_claims)
    duplicate = detect_duplicate_claim(new_claim, claims)
    if duplicate is not None:
        if duplicate.status is ClaimStatus.ACCEPTED:
            return ClaimSupport.SUPPORTED
        return ClaimSupport.DUPLICATE

    has_weak_support = False
    for existing_claim in claims:
        if claims_directly_contradict(new_claim, existing_claim):
            return ClaimSupport.CONTRADICTED
        if claims_have_functional_conflict(new_claim, existing_claim):
            return ClaimSupport.CONTRADICTED
        if claims_are_weakly_related(new_claim, existing_claim):
            has_weak_support = True

    if has_weak_support:
        return ClaimSupport.WEAKLY_SUPPORTED
    return ClaimSupport.UNKNOWN


def find_related_claims(
    new_claim: ClaimTriple,
    existing_claims: Iterable[ClaimTriple],
) -> tuple[ClaimTriple, ...]:
    """Return same-subject claims that could inform support or conflict rules."""

    new_subject, new_relation, _new_object = normalize_claim(new_claim)
    related: list[ClaimTriple] = []
    for existing_claim in existing_claims:
        existing_subject, existing_relation, _existing_object = normalize_claim(existing_claim)
        if existing_subject == new_subject and (
            existing_relation == new_relation
            or frozenset((existing_relation, new_relation)) in DIRECT_CONTRADICTION_RELATIONS
        ):
            related.append(existing_claim)
    return tuple(related)


def create_memory_conflict(
    new_claim: ClaimTriple,
    existing_claim: ClaimTriple,
    *,
    rule: str,
) -> MemoryConflict:
    if rule == "direct_relation_contradiction":
        severity = ConflictSeverity.HIGH
        explanation = (
            "Direct contradiction: same normalized subject and object with "
            "opposing relation pair."
        )
    elif rule == "functional_relation_conflict":
        severity = ConflictSeverity.MEDIUM
        explanation = (
            "Functional relation conflict: same normalized subject and "
            "single-valued relation with different objects."
        )
    else:
        severity = ConflictSeverity.MEDIUM
        explanation = "Rule-based KG contradiction."

    return MemoryConflict(
        new_claim=new_claim,
        existing_claim=existing_claim,
        conflict_type=ConflictType.CONTRADICTION,
        severity=severity,
        confidence=1.0,
        explanation=explanation,
        evidence={
            "rule": rule,
            "new_claim_normalized": normalize_claim(new_claim),
            "existing_claim_normalized": normalize_claim(existing_claim),
        },
    )


def find_claim_conflicts(
    new_claim: ClaimTriple,
    existing_claims: Iterable[ClaimTriple],
) -> tuple[MemoryConflict, ...]:
    """Return deterministic contradiction objects for a new claim."""

    conflicts: list[MemoryConflict] = []
    for existing_claim in existing_claims:
        if claims_directly_contradict(new_claim, existing_claim):
            conflicts.append(
                create_memory_conflict(
                    new_claim,
                    existing_claim,
                    rule="direct_relation_contradiction",
                )
            )
        elif claims_have_functional_conflict(new_claim, existing_claim):
            conflicts.append(
                create_memory_conflict(
                    new_claim,
                    existing_claim,
                    rule="functional_relation_conflict",
                )
            )
    return tuple(conflicts)
