"""Small deterministic metrics for handcrafted PRISM evaluation fixtures."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def _check_value(result: Mapping[str, Any], name: str) -> bool | None:
    checks = result.get("checks", {})
    value = checks.get(name) if isinstance(checks, Mapping) else None
    if value is None:
        return None
    return bool(value)


def _rate(values: Sequence[bool]) -> float | None:
    if not values:
        return None
    return sum(1 for value in values if value) / len(values)


def extraction_success_rate(case_results: Sequence[Mapping[str, Any]]) -> float | None:
    """Fraction of cases meeting `expected.min_claims_extracted`."""

    values = [
        value
        for result in case_results
        if (value := _check_value(result, "extraction_success")) is not None
    ]
    return _rate(values)


def conflict_detection_accuracy(case_results: Sequence[Mapping[str, Any]]) -> float | None:
    """Fraction of cases matching `expected.has_conflict`."""

    values = [
        value
        for result in case_results
        if (value := _check_value(result, "conflict_match")) is not None
    ]
    return _rate(values)


def expected_action_accuracy(case_results: Sequence[Mapping[str, Any]]) -> float | None:
    """Fraction of cases matching `expected.expected_action`."""

    values = [
        value
        for result in case_results
        if (value := _check_value(result, "action_match")) is not None
    ]
    return _rate(values)


def expected_risk_level_accuracy(case_results: Sequence[Mapping[str, Any]]) -> float | None:
    """Fraction of cases matching `expected.expected_risk_level`."""

    values = [
        value
        for result in case_results
        if (value := _check_value(result, "risk_level_match")) is not None
    ]
    return _rate(values)


def false_positive_conflict_count(case_results: Sequence[Mapping[str, Any]]) -> int:
    """Count cases where a conflict was observed but not expected."""

    count = 0
    for result in case_results:
        expected = result.get("expected", {})
        observed = result.get("observed", {})
        if (
            isinstance(expected, Mapping)
            and isinstance(observed, Mapping)
            and expected.get("has_conflict") is False
            and observed.get("has_conflict") is True
        ):
            count += 1
    return count


def false_negative_conflict_count(case_results: Sequence[Mapping[str, Any]]) -> int:
    """Count cases where a conflict was expected but not observed."""

    count = 0
    for result in case_results:
        expected = result.get("expected", {})
        observed = result.get("observed", {})
        if (
            isinstance(expected, Mapping)
            and isinstance(observed, Mapping)
            and expected.get("has_conflict") is True
            and observed.get("has_conflict") is False
        ):
            count += 1
    return count


def compute_eval_metrics(case_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compute lightweight fixture-check metrics.

    These are deterministic sanity metrics for handcrafted cases, not PRISM's
    full benchmark metrics.
    """

    return {
        "case_count": len(case_results),
        "extraction_success_rate": extraction_success_rate(case_results),
        "conflict_detection_accuracy": conflict_detection_accuracy(case_results),
        "expected_action_accuracy": expected_action_accuracy(case_results),
        "expected_risk_level_accuracy": expected_risk_level_accuracy(case_results),
        "false_positive_conflict_count": false_positive_conflict_count(case_results),
        "false_negative_conflict_count": false_negative_conflict_count(case_results),
    }
