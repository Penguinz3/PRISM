"""Fixture loading and validation for deterministic PRISM evaluation."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


REQUIRED_FIXTURE_FIELDS = frozenset({"case_id", "prompt", "answer", "expected"})


def _require_string(fixture: Mapping[str, Any], key: str) -> None:
    value = fixture.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"eval fixture field {key!r} must be a non-empty string")


def validate_eval_fixture(fixture: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and return a shallow fixture copy.

    Validation is intentionally lightweight so handcrafted fixtures stay easy to
    edit. This is not a benchmark schema or a general-purpose validation layer.
    """

    if not isinstance(fixture, Mapping):
        raise ValueError("eval fixture must be a mapping")
    missing = REQUIRED_FIXTURE_FIELDS - set(fixture)
    if missing:
        missing_fields = ", ".join(sorted(missing))
        raise ValueError(f"eval fixture is missing required field(s): {missing_fields}")

    _require_string(fixture, "case_id")
    _require_string(fixture, "prompt")
    _require_string(fixture, "answer")

    expected = fixture.get("expected")
    if not isinstance(expected, Mapping):
        raise ValueError("eval fixture field 'expected' must be a mapping")

    for list_key in ("claims", "memory_claims"):
        if list_key in fixture and not isinstance(fixture[list_key], list):
            raise ValueError(f"eval fixture field {list_key!r} must be a list when present")

    if "min_claims_extracted" in expected:
        min_claims = expected["min_claims_extracted"]
        if not isinstance(min_claims, int) or min_claims < 0:
            raise ValueError("expected.min_claims_extracted must be a non-negative integer")
    if "has_conflict" in expected and not isinstance(expected["has_conflict"], bool):
        raise ValueError("expected.has_conflict must be a boolean")
    for expected_key in (
        "expected_conflict_type",
        "expected_risk_level",
        "expected_action",
    ):
        if expected_key in expected:
            value = expected[expected_key]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"expected.{expected_key} must be a non-empty string")

    return dict(fixture)


def load_eval_fixtures(path: str | Path) -> list[dict[str, Any]]:
    """Load a JSON fixture file and validate each case.

    The file may contain either a top-level list of fixtures or a mapping with a
    `fixtures` list. This keeps examples concise while allowing future metadata.
    """

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, Mapping):
        fixtures = payload.get("fixtures")
    else:
        fixtures = payload
    if not isinstance(fixtures, list):
        raise ValueError("eval fixture file must contain a fixture list")
    return [validate_eval_fixture(fixture) for fixture in fixtures]
