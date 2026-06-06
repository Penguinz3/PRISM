"""Shared PRISM data structures.

These schemas intentionally do not perform model inference, semantic scoring,
claim extraction, graph storage, or advisor calls. They define the traceable
objects those later phases will exchange and log.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping, Sequence
from uuid import uuid4


JsonDict = dict[str, Any]


class RiskLevel(str, Enum):
    """Coarse risk level used by diagnostics and run summaries."""

    UNKNOWN = "unknown"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ClaimStatus(str, Enum):
    """Lifecycle state for a claim triple."""

    PROPOSED = "proposed"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    CONTRADICTED = "contradicted"
    QUARANTINED = "quarantined"


class ConflictType(str, Enum):
    """Conflict categories produced by future memory checks."""

    CONTRADICTION = "contradiction"
    DUPLICATE = "duplicate"
    UNSUPPORTED = "unsupported"
    STALE = "stale"
    DRIFT = "drift"
    OTHER = "other"


class ConflictSeverity(str, Enum):
    """Severity levels for memory conflicts."""

    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


def _new_id() -> str:
    return str(uuid4())


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _datetime_to_str(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _coerce_datetime(value: datetime | str) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value
    if isinstance(value, str):
        normalized = value.replace("Z", "+00:00")
        parsed = datetime.fromisoformat(normalized)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed
    raise TypeError("expected datetime or ISO datetime string")


def _coerce_enum(enum_type: type[Enum], value: Enum | str) -> Enum:
    if isinstance(value, enum_type):
        return value
    return enum_type(value)


def _metadata(value: Mapping[str, Any] | None) -> JsonDict:
    return dict(value or {})


def _validate_non_empty(name: str, value: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"{name} must not be empty")
    return stripped


def _validate_probability(name: str, value: float | None) -> None:
    if value is None:
        return
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0.0 and 1.0")


def _validate_non_negative(name: str, value: float | None) -> None:
    if value is None:
        return
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative")


def _normalize_key_part(value: str) -> str:
    return " ".join(value.lower().split())


@dataclass(frozen=True)
class TokenConfidence:
    """Confidence metadata for one generated token."""

    token: str
    logprob: float | None = None
    probability: float | None = None
    rank: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.token, str) or self.token == "":
            raise ValueError("token must be a non-empty string")
        _validate_probability("probability", self.probability)
        if self.rank is not None and self.rank < 1:
            raise ValueError("rank must be a positive integer")
        object.__setattr__(self, "metadata", _metadata(self.metadata))

    def to_dict(self) -> JsonDict:
        return {
            "token": self.token,
            "logprob": self.logprob,
            "probability": self.probability,
            "rank": self.rank,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TokenConfidence":
        return cls(
            token=data["token"],
            logprob=data.get("logprob"),
            probability=data.get("probability"),
            rank=data.get("rank"),
            metadata=data.get("metadata", {}),
        )


@dataclass(frozen=True)
class GeneratedAnswer:
    """One generated answer and its trace metadata."""

    prompt: str
    text: str
    answer_id: str = field(default_factory=_new_id)
    model_name: str | None = None
    created_at: datetime = field(default_factory=_utc_now)
    sampling_parameters: Mapping[str, Any] = field(default_factory=dict)
    token_confidences: Sequence[TokenConfidence | Mapping[str, Any]] = field(default_factory=tuple)
    mean_logprob: float | None = None
    finish_reason: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "prompt", _validate_non_empty("prompt", self.prompt))
        object.__setattr__(self, "text", _validate_non_empty("text", self.text))
        object.__setattr__(self, "answer_id", _validate_non_empty("answer_id", self.answer_id))
        object.__setattr__(self, "created_at", _coerce_datetime(self.created_at))
        object.__setattr__(self, "sampling_parameters", _metadata(self.sampling_parameters))
        object.__setattr__(self, "metadata", _metadata(self.metadata))
        object.__setattr__(
            self,
            "token_confidences",
            tuple(
                token
                if isinstance(token, TokenConfidence)
                else TokenConfidence.from_dict(token)
                for token in self.token_confidences
            ),
        )

    def to_dict(self) -> JsonDict:
        return {
            "prompt": self.prompt,
            "text": self.text,
            "answer_id": self.answer_id,
            "model_name": self.model_name,
            "created_at": _datetime_to_str(self.created_at),
            "sampling_parameters": dict(self.sampling_parameters),
            "token_confidences": [token.to_dict() for token in self.token_confidences],
            "mean_logprob": self.mean_logprob,
            "finish_reason": self.finish_reason,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GeneratedAnswer":
        return cls(
            prompt=data["prompt"],
            text=data["text"],
            answer_id=data.get("answer_id", _new_id()),
            model_name=data.get("model_name"),
            created_at=_coerce_datetime(data.get("created_at", _utc_now())),
            sampling_parameters=data.get("sampling_parameters", {}),
            token_confidences=data.get("token_confidences", ()),
            mean_logprob=data.get("mean_logprob"),
            finish_reason=data.get("finish_reason"),
            metadata=data.get("metadata", {}),
        )


@dataclass(frozen=True)
class SampleSet:
    """Primary answer plus sampled alternatives for one prompt."""

    prompt: str
    primary_answer: GeneratedAnswer
    samples: Sequence[GeneratedAnswer | Mapping[str, Any]] = field(default_factory=tuple)
    sample_set_id: str = field(default_factory=_new_id)
    created_at: datetime = field(default_factory=_utc_now)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "prompt", _validate_non_empty("prompt", self.prompt))
        object.__setattr__(self, "sample_set_id", _validate_non_empty("sample_set_id", self.sample_set_id))
        object.__setattr__(self, "created_at", _coerce_datetime(self.created_at))
        object.__setattr__(self, "metadata", _metadata(self.metadata))
        if not isinstance(self.primary_answer, GeneratedAnswer):
            object.__setattr__(self, "primary_answer", GeneratedAnswer.from_dict(self.primary_answer))
        object.__setattr__(
            self,
            "samples",
            tuple(
                sample
                if isinstance(sample, GeneratedAnswer)
                else GeneratedAnswer.from_dict(sample)
                for sample in self.samples
            ),
        )

    @property
    def sample_count(self) -> int:
        return len(self.samples)

    @property
    def all_answers(self) -> tuple[GeneratedAnswer, ...]:
        return (self.primary_answer, *self.samples)

    def to_dict(self) -> JsonDict:
        return {
            "prompt": self.prompt,
            "primary_answer": self.primary_answer.to_dict(),
            "samples": [sample.to_dict() for sample in self.samples],
            "sample_set_id": self.sample_set_id,
            "created_at": _datetime_to_str(self.created_at),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SampleSet":
        return cls(
            prompt=data["prompt"],
            primary_answer=GeneratedAnswer.from_dict(data["primary_answer"]),
            samples=data.get("samples", ()),
            sample_set_id=data.get("sample_set_id", _new_id()),
            created_at=_coerce_datetime(data.get("created_at", _utc_now())),
            metadata=data.get("metadata", {}),
        )


@dataclass(frozen=True)
class UncertaintyScores:
    """Risk and uncertainty scores attached to an answer or run."""

    mean_logprob: float | None = None
    mean_token_probability: float | None = None
    self_consistency_disagreement: float | None = None
    semantic_entropy: float | None = None
    semantic_entropy_normalized: float | None = None
    kg_conflict_score: float | None = None
    unsupported_claim_score: float | None = None
    combined_risk_score: float | None = None
    risk_level: RiskLevel | str = RiskLevel.UNKNOWN
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_probability("mean_token_probability", self.mean_token_probability)
        _validate_probability("self_consistency_disagreement", self.self_consistency_disagreement)
        _validate_non_negative("semantic_entropy", self.semantic_entropy)
        _validate_probability("semantic_entropy_normalized", self.semantic_entropy_normalized)
        _validate_probability("kg_conflict_score", self.kg_conflict_score)
        _validate_probability("unsupported_claim_score", self.unsupported_claim_score)
        _validate_probability("combined_risk_score", self.combined_risk_score)
        object.__setattr__(self, "risk_level", _coerce_enum(RiskLevel, self.risk_level))
        object.__setattr__(self, "details", _metadata(self.details))

    @property
    def risk_vector(self) -> JsonDict:
        return {
            "L": self.mean_logprob,
            "S": self.self_consistency_disagreement,
            "H": self.semantic_entropy,
            "K": self.kg_conflict_score,
            "U": self.unsupported_claim_score,
        }

    def to_dict(self) -> JsonDict:
        return {
            "mean_logprob": self.mean_logprob,
            "mean_token_probability": self.mean_token_probability,
            "self_consistency_disagreement": self.self_consistency_disagreement,
            "semantic_entropy": self.semantic_entropy,
            "semantic_entropy_normalized": self.semantic_entropy_normalized,
            "kg_conflict_score": self.kg_conflict_score,
            "unsupported_claim_score": self.unsupported_claim_score,
            "combined_risk_score": self.combined_risk_score,
            "risk_level": self.risk_level.value,
            "details": dict(self.details),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "UncertaintyScores":
        return cls(
            mean_logprob=data.get("mean_logprob"),
            mean_token_probability=data.get("mean_token_probability"),
            self_consistency_disagreement=data.get("self_consistency_disagreement"),
            semantic_entropy=data.get("semantic_entropy"),
            semantic_entropy_normalized=data.get("semantic_entropy_normalized"),
            kg_conflict_score=data.get("kg_conflict_score"),
            unsupported_claim_score=data.get("unsupported_claim_score"),
            combined_risk_score=data.get("combined_risk_score"),
            risk_level=data.get("risk_level", RiskLevel.UNKNOWN),
            details=data.get("details", {}),
        )


@dataclass(frozen=True)
class ClaimTriple:
    """A structured factual claim intended for KG memory."""

    subject: str
    relation: str
    object: str
    confidence: float = 1.0
    source: str | None = None
    timestamp: datetime = field(default_factory=_utc_now)
    status: ClaimStatus | str = ClaimStatus.PROPOSED
    claim_id: str = field(default_factory=_new_id)
    run_id: str | None = None
    turn_id: str | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "subject", _validate_non_empty("subject", self.subject))
        object.__setattr__(self, "relation", _validate_non_empty("relation", self.relation))
        object.__setattr__(self, "object", _validate_non_empty("object", self.object))
        _validate_probability("confidence", self.confidence)
        object.__setattr__(self, "timestamp", _coerce_datetime(self.timestamp))
        object.__setattr__(self, "status", _coerce_enum(ClaimStatus, self.status))
        object.__setattr__(self, "claim_id", _validate_non_empty("claim_id", self.claim_id))
        object.__setattr__(self, "provenance", _metadata(self.provenance))
        object.__setattr__(self, "metadata", _metadata(self.metadata))

    @property
    def canonical_key(self) -> tuple[str, str, str]:
        return (
            _normalize_key_part(self.subject),
            _normalize_key_part(self.relation),
            _normalize_key_part(self.object),
        )

    def to_dict(self) -> JsonDict:
        return {
            "subject": self.subject,
            "relation": self.relation,
            "object": self.object,
            "confidence": self.confidence,
            "source": self.source,
            "timestamp": _datetime_to_str(self.timestamp),
            "status": self.status.value,
            "claim_id": self.claim_id,
            "run_id": self.run_id,
            "turn_id": self.turn_id,
            "provenance": dict(self.provenance),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ClaimTriple":
        return cls(
            subject=data["subject"],
            relation=data["relation"],
            object=data["object"],
            confidence=data.get("confidence", 1.0),
            source=data.get("source"),
            timestamp=_coerce_datetime(data.get("timestamp", _utc_now())),
            status=data.get("status", ClaimStatus.PROPOSED),
            claim_id=data.get("claim_id", _new_id()),
            run_id=data.get("run_id"),
            turn_id=data.get("turn_id"),
            provenance=data.get("provenance", {}),
            metadata=data.get("metadata", {}),
        )


@dataclass(frozen=True)
class MemoryConflict:
    """A detected relationship between a new claim and existing KG memory."""

    new_claim: ClaimTriple
    existing_claim: ClaimTriple
    conflict_type: ConflictType | str
    severity: ConflictSeverity | str = ConflictSeverity.MEDIUM
    confidence: float = 1.0
    explanation: str = ""
    detected_at: datetime = field(default_factory=_utc_now)
    conflict_id: str = field(default_factory=_new_id)
    evidence: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.new_claim, ClaimTriple):
            object.__setattr__(self, "new_claim", ClaimTriple.from_dict(self.new_claim))
        if not isinstance(self.existing_claim, ClaimTriple):
            object.__setattr__(self, "existing_claim", ClaimTriple.from_dict(self.existing_claim))
        object.__setattr__(self, "conflict_type", _coerce_enum(ConflictType, self.conflict_type))
        object.__setattr__(self, "severity", _coerce_enum(ConflictSeverity, self.severity))
        _validate_probability("confidence", self.confidence)
        object.__setattr__(self, "detected_at", _coerce_datetime(self.detected_at))
        object.__setattr__(self, "conflict_id", _validate_non_empty("conflict_id", self.conflict_id))
        object.__setattr__(self, "evidence", _metadata(self.evidence))
        object.__setattr__(self, "metadata", _metadata(self.metadata))

    def to_dict(self) -> JsonDict:
        return {
            "new_claim": self.new_claim.to_dict(),
            "existing_claim": self.existing_claim.to_dict(),
            "conflict_type": self.conflict_type.value,
            "severity": self.severity.value,
            "confidence": self.confidence,
            "explanation": self.explanation,
            "detected_at": _datetime_to_str(self.detected_at),
            "conflict_id": self.conflict_id,
            "evidence": dict(self.evidence),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MemoryConflict":
        return cls(
            new_claim=ClaimTriple.from_dict(data["new_claim"]),
            existing_claim=ClaimTriple.from_dict(data["existing_claim"]),
            conflict_type=data["conflict_type"],
            severity=data.get("severity", ConflictSeverity.MEDIUM),
            confidence=data.get("confidence", 1.0),
            explanation=data.get("explanation", ""),
            detected_at=_coerce_datetime(data.get("detected_at", _utc_now())),
            conflict_id=data.get("conflict_id", _new_id()),
            evidence=data.get("evidence", {}),
            metadata=data.get("metadata", {}),
        )


@dataclass(frozen=True)
class AdvisorDiagnostic:
    """Structured diagnostic packet for a future second-pass advisor."""

    risk_level: RiskLevel | str
    summary: str
    risk_score: float | None = None
    low_confidence_markers: Sequence[str] = field(default_factory=tuple)
    semantic_disagreement_summary: str | None = None
    kg_conflicts: Sequence[MemoryConflict | Mapping[str, Any]] = field(default_factory=tuple)
    claims_safe_to_store: Sequence[ClaimTriple | Mapping[str, Any]] = field(default_factory=tuple)
    claims_requiring_verification: Sequence[ClaimTriple | Mapping[str, Any]] = field(default_factory=tuple)
    suggested_revised_answer_instruction: str | None = None
    diagnostic_id: str = field(default_factory=_new_id)
    created_at: datetime = field(default_factory=_utc_now)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "risk_level", _coerce_enum(RiskLevel, self.risk_level))
        object.__setattr__(self, "summary", _validate_non_empty("summary", self.summary))
        _validate_probability("risk_score", self.risk_score)
        object.__setattr__(self, "low_confidence_markers", tuple(self.low_confidence_markers))
        object.__setattr__(
            self,
            "kg_conflicts",
            tuple(
                conflict
                if isinstance(conflict, MemoryConflict)
                else MemoryConflict.from_dict(conflict)
                for conflict in self.kg_conflicts
            ),
        )
        object.__setattr__(
            self,
            "claims_safe_to_store",
            tuple(
                claim
                if isinstance(claim, ClaimTriple)
                else ClaimTriple.from_dict(claim)
                for claim in self.claims_safe_to_store
            ),
        )
        object.__setattr__(
            self,
            "claims_requiring_verification",
            tuple(
                claim
                if isinstance(claim, ClaimTriple)
                else ClaimTriple.from_dict(claim)
                for claim in self.claims_requiring_verification
            ),
        )
        object.__setattr__(self, "diagnostic_id", _validate_non_empty("diagnostic_id", self.diagnostic_id))
        object.__setattr__(self, "created_at", _coerce_datetime(self.created_at))
        object.__setattr__(self, "metadata", _metadata(self.metadata))

    def to_dict(self) -> JsonDict:
        return {
            "risk_level": self.risk_level.value,
            "summary": self.summary,
            "risk_score": self.risk_score,
            "low_confidence_markers": list(self.low_confidence_markers),
            "semantic_disagreement_summary": self.semantic_disagreement_summary,
            "kg_conflicts": [conflict.to_dict() for conflict in self.kg_conflicts],
            "claims_safe_to_store": [claim.to_dict() for claim in self.claims_safe_to_store],
            "claims_requiring_verification": [
                claim.to_dict() for claim in self.claims_requiring_verification
            ],
            "suggested_revised_answer_instruction": self.suggested_revised_answer_instruction,
            "diagnostic_id": self.diagnostic_id,
            "created_at": _datetime_to_str(self.created_at),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AdvisorDiagnostic":
        return cls(
            risk_level=data["risk_level"],
            summary=data["summary"],
            risk_score=data.get("risk_score"),
            low_confidence_markers=data.get("low_confidence_markers", ()),
            semantic_disagreement_summary=data.get("semantic_disagreement_summary"),
            kg_conflicts=data.get("kg_conflicts", ()),
            claims_safe_to_store=data.get("claims_safe_to_store", ()),
            claims_requiring_verification=data.get("claims_requiring_verification", ()),
            suggested_revised_answer_instruction=data.get("suggested_revised_answer_instruction"),
            diagnostic_id=data.get("diagnostic_id", _new_id()),
            created_at=_coerce_datetime(data.get("created_at", _utc_now())),
            metadata=data.get("metadata", {}),
        )


@dataclass(frozen=True)
class PRISMRunResult:
    """Full structured result for one PRISM run."""

    prompt: str
    primary_answer: GeneratedAnswer
    run_id: str = field(default_factory=_new_id)
    sample_set: SampleSet | Mapping[str, Any] | None = None
    uncertainty_scores: UncertaintyScores | Mapping[str, Any] = field(default_factory=UncertaintyScores)
    extracted_claims: Sequence[ClaimTriple | Mapping[str, Any]] = field(default_factory=tuple)
    memory_conflicts: Sequence[MemoryConflict | Mapping[str, Any]] = field(default_factory=tuple)
    advisor_diagnostic: AdvisorDiagnostic | Mapping[str, Any] | None = None
    model_info: Mapping[str, Any] = field(default_factory=dict)
    config: Mapping[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=_utc_now)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "prompt", _validate_non_empty("prompt", self.prompt))
        object.__setattr__(self, "run_id", _validate_non_empty("run_id", self.run_id))
        if not isinstance(self.primary_answer, GeneratedAnswer):
            object.__setattr__(self, "primary_answer", GeneratedAnswer.from_dict(self.primary_answer))
        if self.sample_set is not None and not isinstance(self.sample_set, SampleSet):
            object.__setattr__(self, "sample_set", SampleSet.from_dict(self.sample_set))
        if not isinstance(self.uncertainty_scores, UncertaintyScores):
            object.__setattr__(self, "uncertainty_scores", UncertaintyScores.from_dict(self.uncertainty_scores))
        object.__setattr__(
            self,
            "extracted_claims",
            tuple(
                claim
                if isinstance(claim, ClaimTriple)
                else ClaimTriple.from_dict(claim)
                for claim in self.extracted_claims
            ),
        )
        object.__setattr__(
            self,
            "memory_conflicts",
            tuple(
                conflict
                if isinstance(conflict, MemoryConflict)
                else MemoryConflict.from_dict(conflict)
                for conflict in self.memory_conflicts
            ),
        )
        if self.advisor_diagnostic is not None and not isinstance(
            self.advisor_diagnostic, AdvisorDiagnostic
        ):
            object.__setattr__(
                self,
                "advisor_diagnostic",
                AdvisorDiagnostic.from_dict(self.advisor_diagnostic),
            )
        object.__setattr__(self, "model_info", _metadata(self.model_info))
        object.__setattr__(self, "config", _metadata(self.config))
        object.__setattr__(self, "timestamp", _coerce_datetime(self.timestamp))
        object.__setattr__(self, "metadata", _metadata(self.metadata))

    def to_dict(self) -> JsonDict:
        return {
            "prompt": self.prompt,
            "primary_answer": self.primary_answer.to_dict(),
            "run_id": self.run_id,
            "sample_set": self.sample_set.to_dict() if self.sample_set else None,
            "uncertainty_scores": self.uncertainty_scores.to_dict(),
            "extracted_claims": [claim.to_dict() for claim in self.extracted_claims],
            "memory_conflicts": [conflict.to_dict() for conflict in self.memory_conflicts],
            "advisor_diagnostic": (
                self.advisor_diagnostic.to_dict() if self.advisor_diagnostic else None
            ),
            "model_info": dict(self.model_info),
            "config": dict(self.config),
            "timestamp": _datetime_to_str(self.timestamp),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PRISMRunResult":
        advisor_data = data.get("advisor_diagnostic")
        sample_set_data = data.get("sample_set")
        return cls(
            prompt=data["prompt"],
            primary_answer=GeneratedAnswer.from_dict(data["primary_answer"]),
            run_id=data.get("run_id", _new_id()),
            sample_set=SampleSet.from_dict(sample_set_data) if sample_set_data else None,
            uncertainty_scores=UncertaintyScores.from_dict(data.get("uncertainty_scores", {})),
            extracted_claims=data.get("extracted_claims", ()),
            memory_conflicts=data.get("memory_conflicts", ()),
            advisor_diagnostic=AdvisorDiagnostic.from_dict(advisor_data) if advisor_data else None,
            model_info=data.get("model_info", {}),
            config=data.get("config", {}),
            timestamp=_coerce_datetime(data.get("timestamp", _utc_now())),
            metadata=data.get("metadata", {}),
        )
