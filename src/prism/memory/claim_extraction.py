"""Deterministic claim extraction stubs for PRISM.

This module supports simple pipe-delimited triples and a small set of explicit
natural-language patterns. It is not robust NLP and does not call an LLM.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Iterable

from prism.schemas import ClaimStatus, ClaimTriple


class ClaimExtractionError(ValueError):
    """Raised when strict extraction finds no claims."""


DEFAULT_EXTRACTION_CONFIDENCE = 0.5

_PIPE_SPLIT_RE = re.compile(r"\s*\|\s*")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("does_not_have", re.compile(r"^(?P<subject>.+?)\s+does\s+not\s+have\s+(?P<object>.+)$", re.I)),
    ("is_not", re.compile(r"^(?P<subject>.+?)\s+is\s+not\s+(?P<object>.+)$", re.I)),
    ("supports", re.compile(r"^(?P<subject>.+?)\s+supports\s+(?P<object>.+)$", re.I)),
    ("contradicts", re.compile(r"^(?P<subject>.+?)\s+contradicts\s+(?P<object>.+)$", re.I)),
    ("uses", re.compile(r"^(?P<subject>.+?)\s+uses\s+(?P<object>.+)$", re.I)),
    ("has", re.compile(r"^(?P<subject>.+?)\s+has\s+(?P<object>.+)$", re.I)),
    ("is", re.compile(r"^(?P<subject>.+?)\s+is\s+(?P<object>.+)$", re.I)),
)


def _coerce_timestamp(value: str | datetime | None) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _clean_part(value: str) -> str:
    return value.strip(" \t\r\n\"'.,;:")


def _iter_segments(text: str) -> Iterable[str]:
    for raw_segment in _SENTENCE_SPLIT_RE.split(text):
        segment = raw_segment.strip()
        if not segment:
            continue
        yield segment


def _claim(
    *,
    subject: str,
    relation: str,
    object_: str,
    confidence: float,
    source: str,
    timestamp: datetime,
    run_id: str | None,
    turn_id: str | None,
    provenance_text: str,
) -> ClaimTriple:
    return ClaimTriple(
        subject=_clean_part(subject),
        relation=_clean_part(relation),
        object=_clean_part(object_),
        confidence=confidence,
        source=source,
        timestamp=timestamp,
        status=ClaimStatus.PROPOSED,
        run_id=run_id,
        turn_id=turn_id,
        provenance={
            "extractor": "deterministic_stub",
            "text": provenance_text,
        },
    )


def _extract_pipe_claim(segment: str, **kwargs) -> ClaimTriple | None:
    parts = [_clean_part(part) for part in _PIPE_SPLIT_RE.split(segment)]
    if len(parts) != 3 or not all(parts):
        return None
    return _claim(
        subject=parts[0],
        relation=parts[1],
        object_=parts[2],
        provenance_text=segment,
        **kwargs,
    )


def _extract_pattern_claim(segment: str, **kwargs) -> ClaimTriple | None:
    for relation, pattern in _PATTERNS:
        match = pattern.match(segment)
        if not match:
            continue
        return _claim(
            subject=match.group("subject"),
            relation=relation,
            object_=match.group("object"),
            provenance_text=segment,
            **kwargs,
        )
    return None


def extract_claims(
    text: str,
    source: str = "auto_extraction",
    run_id: str | None = None,
    turn_id: str | None = None,
    timestamp: str | datetime | None = None,
    confidence: float = DEFAULT_EXTRACTION_CONFIDENCE,
    strict: bool = False,
) -> list[ClaimTriple]:
    """Extract simple claim triples from text.

    Supported forms:

    - `subject | relation | object`
    - `subject is object`
    - `subject is not object`
    - `subject has object`
    - `subject does not have object`
    - `subject uses object`
    - `subject supports object`
    - `subject contradicts object`
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    extracted_at = _coerce_timestamp(timestamp)
    common = {
        "confidence": confidence,
        "source": source,
        "timestamp": extracted_at,
        "run_id": run_id,
        "turn_id": turn_id,
    }

    claims: list[ClaimTriple] = []
    for segment in _iter_segments(text):
        claim = _extract_pipe_claim(segment, **common)
        if claim is None:
            claim = _extract_pattern_claim(segment, **common)
        if claim is not None:
            claims.append(claim)

    if strict and not claims:
        raise ClaimExtractionError("no claims extracted")
    return claims
