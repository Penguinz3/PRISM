from prism.evaluation.metrics import (
    compute_eval_metrics,
    conflict_detection_accuracy,
    expected_action_accuracy,
    expected_risk_level_accuracy,
    extraction_success_rate,
    false_negative_conflict_count,
    false_positive_conflict_count,
)


def case_result(
    *,
    extraction_success: bool = True,
    conflict_match: bool = True,
    action_match: bool = True,
    risk_level_match: bool = True,
    expected_conflict: bool = False,
    observed_conflict: bool = False,
) -> dict:
    return {
        "expected": {"has_conflict": expected_conflict},
        "observed": {"has_conflict": observed_conflict},
        "checks": {
            "extraction_success": extraction_success,
            "conflict_match": conflict_match,
            "action_match": action_match,
            "risk_level_match": risk_level_match,
        },
    }


def test_metric_computation() -> None:
    results = [
        case_result(),
        case_result(extraction_success=False, action_match=False, expected_conflict=True),
        case_result(conflict_match=False, risk_level_match=False, observed_conflict=True),
    ]

    assert extraction_success_rate(results) == 2 / 3
    assert conflict_detection_accuracy(results) == 2 / 3
    assert expected_action_accuracy(results) == 2 / 3
    assert expected_risk_level_accuracy(results) == 2 / 3

    metrics = compute_eval_metrics(results)

    assert metrics["case_count"] == 3
    assert metrics["extraction_success_rate"] == 2 / 3
    assert metrics["conflict_detection_accuracy"] == 2 / 3


def test_false_positive_and_false_negative_conflict_counts() -> None:
    results = [
        case_result(expected_conflict=False, observed_conflict=True),
        case_result(expected_conflict=True, observed_conflict=False),
        case_result(expected_conflict=True, observed_conflict=True),
    ]

    assert false_positive_conflict_count(results) == 1
    assert false_negative_conflict_count(results) == 1


def test_metric_rate_returns_none_when_no_applicable_cases() -> None:
    results = [{"expected": {}, "observed": {}, "checks": {"action_match": None}}]

    assert expected_action_accuracy(results) is None
