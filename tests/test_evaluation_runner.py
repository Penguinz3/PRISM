import json
from pathlib import Path

import pytest

from prism.evaluation.fixtures import load_eval_fixtures, validate_eval_fixture
from prism.evaluation.runner import run_deterministic_eval


EXAMPLE_FIXTURES = Path(__file__).resolve().parents[1] / "examples" / "eval_fixtures.json"


def by_case(report: dict, case_id: str) -> dict:
    return next(result for result in report["case_results"] if result["case_id"] == case_id)


def test_fixture_loading_from_example_file() -> None:
    fixtures = load_eval_fixtures(EXAMPLE_FIXTURES)

    assert len(fixtures) >= 6
    assert fixtures[0]["case_id"] == "safe_supported_use"


def test_fixture_validation_rejects_missing_required_fields() -> None:
    with pytest.raises(ValueError, match="missing required"):
        validate_eval_fixture({"case_id": "broken"})


def test_runner_handles_no_conflict_case() -> None:
    report = run_deterministic_eval(load_eval_fixtures(EXAMPLE_FIXTURES))
    result = by_case(report, "safe_supported_use")

    assert result["observed"]["has_conflict"] is False
    assert result["observed"]["recommended_action"] == "trust"
    assert result["checks"]["conflict_match"] is True
    assert result["checks"]["risk_level_match"] is True


def test_runner_handles_direct_contradiction_case() -> None:
    report = run_deterministic_eval(load_eval_fixtures(EXAMPLE_FIXTURES))
    result = by_case(report, "direct_is_not_contradiction")

    assert result["observed"]["has_conflict"] is True
    assert "direct_relation_contradiction" in result["observed"]["conflict_rules"]
    assert result["checks"]["conflict_type_match"] is True
    assert result["observed"]["recommended_action"] == "reject_or_do_not_store"


def test_runner_handles_functional_relation_conflict_case() -> None:
    report = run_deterministic_eval(load_eval_fixtures(EXAMPLE_FIXTURES))
    result = by_case(report, "functional_default_model_conflict")

    assert result["observed"]["has_conflict"] is True
    assert "functional_relation_conflict" in result["observed"]["conflict_rules"]
    assert result["checks"]["conflict_type_match"] is True


def test_runner_handles_extraction_empty_case() -> None:
    report = run_deterministic_eval(load_eval_fixtures(EXAMPLE_FIXTURES))
    result = by_case(report, "no_extractable_claims")

    assert result["observed"]["claims_extracted_count"] == 0
    assert result["observed"]["has_conflict"] is False
    assert result["checks"]["extraction_success"] is True


def test_runner_writes_json_report(tmp_path) -> None:
    report = run_deterministic_eval(load_eval_fixtures(EXAMPLE_FIXTURES), output_dir=tmp_path)
    report_path = Path(report["report_path"])

    assert report_path.parent.parent == tmp_path
    assert report_path.exists()

    payload = json.loads(report_path.read_text(encoding="utf-8"))

    assert payload["schema_version"] == 1
    assert payload["metrics"]["case_count"] >= 6
    assert payload["report_path"] == str(report_path)


def test_runner_computes_expected_aggregate_metrics() -> None:
    report = run_deterministic_eval(EXAMPLE_FIXTURES)

    assert report["metrics"]["case_count"] >= 6
    assert report["metrics"]["conflict_detection_accuracy"] == 1.0
    assert report["metrics"]["expected_action_accuracy"] == 1.0
    assert report["metrics"]["expected_risk_level_accuracy"] == 1.0
