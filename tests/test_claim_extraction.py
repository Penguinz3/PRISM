from datetime import datetime, timezone

import pytest

from prism import ClaimStatus
from prism.memory import ClaimExtractionError, extract_claims


NOW = datetime(2026, 6, 6, 22, 0, tzinfo=timezone.utc)


def test_structured_pipe_delimited_triple_extraction() -> None:
    claims = extract_claims(
        "PRISM | default_model | Qwen2.5-1.5B-Instruct\n"
        "PRISM | uses_marker | semantic_entropy"
    )

    assert len(claims) == 2
    assert claims[0].subject == "PRISM"
    assert claims[0].relation == "default_model"
    assert claims[0].object == "Qwen2.5-1.5B-Instruct"
    assert claims[1].relation == "uses_marker"


def test_is_pattern() -> None:
    claims = extract_claims("PRISM is a reliability prototype.")

    assert len(claims) == 1
    assert claims[0].relation == "is"
    assert claims[0].subject == "PRISM"
    assert claims[0].object == "a reliability prototype"


def test_is_not_pattern() -> None:
    claims = extract_claims("PRISM is not a production system.")

    assert claims[0].relation == "is_not"
    assert claims[0].object == "a production system"


def test_has_pattern() -> None:
    claims = extract_claims("PRISM has KG memory.")

    assert claims[0].relation == "has"
    assert claims[0].object == "KG memory"


def test_does_not_have_pattern() -> None:
    claims = extract_claims("PRISM does not have model inference.")

    assert claims[0].relation == "does_not_have"
    assert claims[0].object == "model inference"


def test_uses_pattern() -> None:
    claims = extract_claims("PRISM uses semantic entropy.")

    assert claims[0].relation == "uses"
    assert claims[0].object == "semantic entropy"


def test_supports_and_contradicts_patterns() -> None:
    claims = extract_claims("PRISM supports KG memory. PRISM contradicts stale memory.")

    assert [claim.relation for claim in claims] == ["supports", "contradicts"]
    assert [claim.object for claim in claims] == ["KG memory", "stale memory"]


def test_no_claims_returns_empty_list() -> None:
    assert extract_claims("No structured fact here") == []


def test_strict_mode_raises_when_no_claims() -> None:
    with pytest.raises(ClaimExtractionError):
        extract_claims("No structured fact here", strict=True)


def test_provenance_fields_are_preserved() -> None:
    claims = extract_claims(
        "PRISM uses semantic entropy.",
        source="answer_text",
        run_id="run-1",
        turn_id="turn-1",
        timestamp=NOW,
    )

    claim = claims[0]
    assert claim.source == "answer_text"
    assert claim.run_id == "run-1"
    assert claim.turn_id == "turn-1"
    assert claim.timestamp == NOW
    assert claim.provenance["extractor"] == "deterministic_stub"
    assert claim.provenance["text"] == "PRISM uses semantic entropy."


def test_default_confidence_and_status_behavior() -> None:
    claim = extract_claims("PRISM has KG memory.")[0]

    assert claim.confidence == 0.5
    assert claim.status is ClaimStatus.PROPOSED


def test_mixed_text_and_pipe_triple_preserves_dotted_model_name() -> None:
    claims = extract_claims(
        "PRISM is prototype. PRISM | default_model | Qwen2.5-1.5B-Instruct. "
        "PRISM has KG memory."
    )

    assert [claim.relation for claim in claims] == ["is", "default_model", "has"]
    assert claims[1].object == "Qwen2.5-1.5B-Instruct"
