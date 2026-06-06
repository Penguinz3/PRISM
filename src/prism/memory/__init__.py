"""Knowledge graph memory utilities for PRISM."""

from prism.memory.claim_extraction import (
    DEFAULT_EXTRACTION_CONFIDENCE,
    ClaimExtractionError,
    extract_claims,
)
from prism.memory.conflict_detection import (
    FUNCTIONAL_RELATIONS,
    ClaimSupport,
    classify_claim_support,
    create_memory_conflict,
    detect_duplicate_claim,
    find_claim_conflicts,
    find_related_claims,
    normalize_claim,
    normalize_relation,
    normalize_text,
)
from prism.memory.graph_store import MemoryGraphStore

__all__ = [
    "DEFAULT_EXTRACTION_CONFIDENCE",
    "ClaimExtractionError",
    "FUNCTIONAL_RELATIONS",
    "ClaimSupport",
    "MemoryGraphStore",
    "classify_claim_support",
    "create_memory_conflict",
    "detect_duplicate_claim",
    "extract_claims",
    "find_claim_conflicts",
    "find_related_claims",
    "normalize_claim",
    "normalize_relation",
    "normalize_text",
]
