"""Deterministic evaluation helpers for PRISM fixture checks."""

from prism.evaluation.fixtures import load_eval_fixtures, validate_eval_fixture
from prism.evaluation.metrics import (
    compute_eval_metrics,
    conflict_detection_accuracy,
    expected_action_accuracy,
    expected_risk_level_accuracy,
    extraction_success_rate,
    false_negative_conflict_count,
    false_positive_conflict_count,
)

__all__ = [
    "compute_eval_metrics",
    "conflict_detection_accuracy",
    "expected_action_accuracy",
    "expected_risk_level_accuracy",
    "extraction_success_rate",
    "false_negative_conflict_count",
    "false_positive_conflict_count",
    "load_eval_fixtures",
    "run_deterministic_eval",
    "validate_eval_fixture",
]


def __getattr__(name: str):
    if name == "run_deterministic_eval":
        from prism.evaluation.runner import run_deterministic_eval

        return run_deterministic_eval
    raise AttributeError(name)
