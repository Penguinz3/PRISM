"""Lightweight JSON-backed claim memory for PRISM.

Phase 2 intentionally keeps the memory layer small: claims are stored in memory,
duplicates are detected by normalized subject/relation/object, and persistence is
handled through JSON export/import. There is no graph database, model inference,
or conflict-detection logic here.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Mapping

from prism.schemas import ClaimStatus, ClaimTriple, JsonDict


ClaimInput = ClaimTriple | Mapping[str, Any]


def _normalize(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("lookup value must be a string")
    stripped = " ".join(value.lower().split())
    if not stripped:
        raise ValueError("lookup value must not be empty")
    return stripped


def _coerce_claim(claim: ClaimInput) -> ClaimTriple:
    if isinstance(claim, ClaimTriple):
        return claim
    return ClaimTriple.from_dict(claim)


class MemoryGraphStore:
    """In-memory claim store with deterministic JSON persistence.

    Duplicate handling is deliberately conservative: adding a claim whose
    normalized `(subject, relation, object)` key already exists returns the
    existing claim and leaves stored memory unchanged.
    """

    schema_version = 1

    def __init__(self, claims: Iterable[ClaimInput] | None = None) -> None:
        self._claims_by_key: dict[tuple[str, str, str], ClaimTriple] = {}
        self._keys_by_claim_id: dict[str, tuple[str, str, str]] = {}
        for claim in claims or ():
            self.add_claim(claim)

    def __len__(self) -> int:
        return len(self._claims_by_key)

    @property
    def claims(self) -> tuple[ClaimTriple, ...]:
        """Return claims in deterministic insertion order."""

        return tuple(self._claims_by_key.values())

    @property
    def is_empty(self) -> bool:
        return len(self) == 0

    def add_claim(self, claim: ClaimInput) -> ClaimTriple:
        """Add a claim or return the existing duplicate without overwriting it."""

        claim = _coerce_claim(claim)
        key = claim.canonical_key
        existing = self._claims_by_key.get(key)
        if existing is not None:
            return existing

        existing_key_for_id = self._keys_by_claim_id.get(claim.claim_id)
        if existing_key_for_id is not None and existing_key_for_id != key:
            raise ValueError(f"claim_id already exists with a different claim: {claim.claim_id}")

        self._claims_by_key[key] = claim
        self._keys_by_claim_id[claim.claim_id] = key
        return claim

    def find_duplicate(self, claim: ClaimInput) -> ClaimTriple | None:
        """Return the stored duplicate for a claim, if present."""

        claim = _coerce_claim(claim)
        return self._claims_by_key.get(claim.canonical_key)

    def get_claims_by_entity(self, entity: str) -> tuple[ClaimTriple, ...]:
        """Return claims where the entity appears as subject or object."""

        normalized = _normalize(entity)
        return tuple(
            claim
            for claim in self.claims
            if claim.canonical_key[0] == normalized or claim.canonical_key[2] == normalized
        )

    def get_claims_about_entity(self, entity: str) -> tuple[ClaimTriple, ...]:
        """Alias for `get_claims_by_entity`."""

        return self.get_claims_by_entity(entity)

    def get_claims_by_subject(self, subject: str) -> tuple[ClaimTriple, ...]:
        normalized = _normalize(subject)
        return tuple(claim for claim in self.claims if claim.canonical_key[0] == normalized)

    def get_claims_by_relation(self, relation: str) -> tuple[ClaimTriple, ...]:
        normalized = _normalize(relation)
        return tuple(claim for claim in self.claims if claim.canonical_key[1] == normalized)

    def find_related_claims(self, claim: ClaimInput) -> tuple[ClaimTriple, ...]:
        """Return claims related by deterministic conflict-detection rules."""

        from prism.memory.conflict_detection import find_related_claims

        return find_related_claims(_coerce_claim(claim), self.claims)

    def check_claim_support(self, claim: ClaimInput):
        """Classify support for a claim against current memory."""

        from prism.memory.conflict_detection import classify_claim_support

        return classify_claim_support(_coerce_claim(claim), self.claims)

    def find_conflicts(self, claim: ClaimInput):
        """Return rule-based KG conflicts for a claim against current memory."""

        from prism.memory.conflict_detection import find_claim_conflicts

        return find_claim_conflicts(_coerce_claim(claim), self.claims)

    def update_claim_status(self, claim_id: str, status: ClaimStatus | str) -> ClaimTriple:
        """Update one claim's status by id and return the updated claim."""

        if not isinstance(claim_id, str) or not claim_id.strip():
            raise ValueError("claim_id must be a non-empty string")
        key = self._keys_by_claim_id.get(claim_id)
        if key is None:
            raise KeyError(f"claim_id not found: {claim_id}")

        updated = replace(self._claims_by_key[key], status=ClaimStatus(status))
        self._claims_by_key[key] = updated
        return updated

    def to_dict(self) -> JsonDict:
        return {
            "schema_version": self.schema_version,
            "claims": [claim.to_dict() for claim in self.claims],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | list[Mapping[str, Any]]) -> "MemoryGraphStore":
        if isinstance(data, list):
            claims = data
        else:
            claims = data.get("claims", ())
        return cls(claims=claims)

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "MemoryGraphStore":
        return cls.from_dict(json.loads(text))

    def export_json(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def import_json(cls, path: str | Path) -> "MemoryGraphStore":
        source = Path(path)
        return cls.from_json(source.read_text(encoding="utf-8"))
