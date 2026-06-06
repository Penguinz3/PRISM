import json
from datetime import datetime, timezone

from prism import GeneratedAnswer, PRISMRunResult, UncertaintyScores
from prism.logging import RunLogger, build_run_log_payload


NOW = datetime(2026, 6, 6, 21, 0, tzinfo=timezone.utc)


def make_result() -> PRISMRunResult:
    return PRISMRunResult(
        prompt="What is PRISM?",
        primary_answer=GeneratedAnswer(prompt="What is PRISM?", text="A reliability system."),
        run_id="run-test",
        uncertainty_scores=UncertaintyScores(combined_risk_score=0.2),
        timestamp=NOW,
        config={"thresholds": {"risk_low": 0.33, "risk_high": 0.66}},
        metadata={
            "reliability_score": 0.8,
            "kg_support_classifications": {"claim-1": "supported"},
            "thresholds": {"risk_low": 0.33, "risk_high": 0.66},
        },
    )


def test_run_logger_creates_dated_directory(tmp_path) -> None:
    logger = RunLogger(tmp_path)
    path = logger.write(make_result())

    assert path.parent == tmp_path / "2026-06-06"
    assert path.exists()


def test_run_logger_writes_valid_json(tmp_path) -> None:
    logger = RunLogger(tmp_path)
    path = logger.write(make_result())

    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["run_id"] == "run-test"
    assert payload["prompt"] == "What is PRISM?"
    assert payload["answer"] == "A reliability system."
    assert payload["risk_score"] == 0.2
    assert payload["reliability_score"] == 0.8
    assert payload["kg_support_classifications"] == {"claim-1": "supported"}
    assert payload["package"]["name"] == "prism-reliability"


def test_build_run_log_payload_includes_required_sections() -> None:
    payload = build_run_log_payload(make_result())

    assert "provided_claims" in payload
    assert "uncertainty_scores" in payload
    assert "kg_conflicts" in payload
    assert "advisor_diagnostic" in payload
    assert "config" in payload
