"""Runner for deterministic PRISM evaluation fixtures."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from prism.evaluation.fixtures import load_eval_fixtures, validate_eval_fixture
from prism.evaluation.metrics import compute_eval_metrics
from prism.memory import MemoryGraphStore
from prism.pipeline import run_prism_analysis
from prism.schemas import ClaimTriple, PRISMRunResult, UncertaintyScores


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _safe_timestamp(value: datetime) -> str:
    return value.strftime("%Y%m%dT%H%M%S%fZ")


def _claims_from_payload(items: Sequence[ClaimTriple | Mapping[str, Any]] | None) -> list[ClaimTriple]:
    return [
        item if isinstance(item, ClaimTriple) else ClaimTriple.from_dict(item)
        for item in (items or [])
    ]


def _uncertainty_from_fixture(fixture: Mapping[str, Any]) -> UncertaintyScores | Mapping[str, Any] | None:
    data = fixture.get("uncertainty_scores")
    if data is None:
        return None
    if any(
        key in data
        for key in {
            "mean_logprob",
            "mean_token_probability",
            "self_consistency_disagreement",
            "semantic_entropy_normalized",
            "combined_risk_score",
        }
    ):
        return UncertaintyScores.from_dict(data)
    return data


def _conflict_rules(result: PRISMRunResult) -> list[str]:
    return [
        str(conflict.evidence.get("rule"))
        for conflict in result.memory_conflicts
        if conflict.evidence.get("rule")
    ]


def _observed_from_result(result: PRISMRunResult) -> dict[str, Any]:
    diagnostic = result.advisor_diagnostic
    return {
        "claims_extracted_count": len(result.extracted_claims),
        "has_conflict": bool(result.memory_conflicts),
        "conflict_types": [
            conflict.conflict_type.value for conflict in result.memory_conflicts
        ],
        "conflict_rules": _conflict_rules(result),
        "risk_level": diagnostic.risk_level.value if diagnostic else None,
        "recommended_action": (
            diagnostic.metadata.get("recommended_action") if diagnostic else None
        ),
        "kg_support_classifications": result.metadata.get("kg_support_classifications", {}),
        "risk_score": result.metadata.get("risk_score"),
        "reliability_score": result.metadata.get("reliability_score"),
    }


def _checks(expected: Mapping[str, Any], observed: Mapping[str, Any]) -> dict[str, bool | None]:
    expected_conflict_type = expected.get("expected_conflict_type")
    conflict_labels = set(observed.get("conflict_types", ())) | set(
        observed.get("conflict_rules", ())
    )
    return {
        "extraction_success": (
            None
            if "min_claims_extracted" not in expected
            else observed["claims_extracted_count"] >= expected["min_claims_extracted"]
        ),
        "conflict_match": (
            None
            if "has_conflict" not in expected
            else observed["has_conflict"] is expected["has_conflict"]
        ),
        "conflict_type_match": (
            None
            if expected_conflict_type is None
            else expected_conflict_type in conflict_labels
        ),
        "risk_level_match": (
            None
            if "expected_risk_level" not in expected
            else observed["risk_level"] == expected["expected_risk_level"]
        ),
        "action_match": (
            None
            if "expected_action" not in expected
            else observed["recommended_action"] == expected["expected_action"]
        ),
    }


def _run_case(fixture: Mapping[str, Any]) -> dict[str, Any]:
    validated = validate_eval_fixture(fixture)
    memory_store = MemoryGraphStore(_claims_from_payload(validated.get("memory_claims")))
    claims = _claims_from_payload(validated.get("claims"))
    auto_extract_claims = bool(
        validated.get("auto_extract_claims", True if not claims else False)
    )
    result = run_prism_analysis(
        prompt=validated["prompt"],
        answer=validated["answer"],
        claims=claims,
        memory_store=memory_store,
        uncertainty_scores=_uncertainty_from_fixture(validated),
        write_log=False,
        auto_extract_claims=auto_extract_claims,
        config={
            "eval_case_id": validated["case_id"],
            "eval_mode": "deterministic_fixture",
        },
    )
    observed = _observed_from_result(result)
    expected = dict(validated["expected"])
    return {
        "case_id": validated["case_id"],
        "expected": expected,
        "observed": observed,
        "checks": _checks(expected, observed),
        "prism_run_result": result.to_dict(),
    }


def _eval_report_path(output_dir: str | Path, timestamp: datetime) -> Path:
    return Path(output_dir) / timestamp.date().isoformat() / f"eval_{_safe_timestamp(timestamp)}.json"


def _write_eval_report(report: Mapping[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return path


def run_deterministic_eval(
    fixtures: Sequence[Mapping[str, Any]] | str | Path,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run deterministic PRISM fixtures and optionally write a JSON report."""

    loaded_fixtures = load_eval_fixtures(fixtures) if isinstance(fixtures, (str, Path)) else list(fixtures)
    timestamp = _utc_now()
    case_results = [_run_case(fixture) for fixture in loaded_fixtures]
    report: dict[str, Any] = {
        "schema_version": 1,
        "timestamp": timestamp.isoformat(),
        "mode": "deterministic_fixture_eval",
        "metrics": compute_eval_metrics(case_results),
        "case_results": case_results,
        "report_path": None,
    }
    if output_dir is not None:
        path = _eval_report_path(output_dir, timestamp)
        report["report_path"] = str(path)
        _write_eval_report(report, path)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run deterministic PRISM evaluation fixtures.")
    parser.add_argument("--input", required=True, help="Path to an eval fixture JSON file.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional eval report root directory, e.g. logs/evals.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    report = run_deterministic_eval(args.input, output_dir=args.output_dir)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
