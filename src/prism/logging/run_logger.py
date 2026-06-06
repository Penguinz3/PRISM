"""JSON run logger for PRISM analysis results."""

from __future__ import annotations

import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from prism.schemas import PRISMRunResult


def _package_version() -> str:
    try:
        return version("prism-reliability")
    except PackageNotFoundError:
        return "0.1.0"


def _safe_timestamp(value) -> str:
    return value.strftime("%Y%m%dT%H%M%S%fZ")


def build_run_log_payload(result: PRISMRunResult) -> dict:
    """Build a stable JSON payload without reading environment variables."""

    result_payload = result.to_dict()
    metadata = dict(result.metadata)
    primary_answer = result_payload["primary_answer"]
    sample_set = result_payload.get("sample_set") or {}

    return {
        "schema_version": 1,
        "package": {
            "name": "prism-reliability",
            "version": _package_version(),
        },
        "run_id": result.run_id,
        "timestamp": result_payload["timestamp"],
        "prompt": result.prompt,
        "answer": primary_answer["text"],
        "backend_name": metadata.get("backend_name") or result.model_info.get("backend_name"),
        "model_name": primary_answer.get("model_name") or result.model_info.get("model_name"),
        "generation_config": metadata.get("generation_config", {}),
        "generated_answer": primary_answer,
        "samples": sample_set.get("samples", []),
        "token_confidences": primary_answer.get("token_confidences", []),
        "provided_claims": result_payload["extracted_claims"],
        "uncertainty_scores": result_payload["uncertainty_scores"],
        "reliability_score": metadata.get("reliability_score"),
        "risk_score": result.uncertainty_scores.combined_risk_score,
        "kg_support_classifications": metadata.get("kg_support_classifications", {}),
        "kg_conflicts": result_payload["memory_conflicts"],
        "advisor_diagnostic": result_payload["advisor_diagnostic"],
        "config": result_payload["config"],
        "thresholds": metadata.get("thresholds", {}),
        "prism_run_result": result_payload,
    }


class RunLogger:
    """Write PRISM run logs under `logs/runs/YYYY-MM-DD/`."""

    def __init__(self, output_dir: str | Path = "logs/runs") -> None:
        self.output_dir = Path(output_dir)

    def path_for_result(self, result: PRISMRunResult) -> Path:
        date_dir = result.timestamp.date().isoformat()
        filename = f"run_{_safe_timestamp(result.timestamp)}_{result.run_id}.json"
        return self.output_dir / date_dir / filename

    def write(self, result: PRISMRunResult) -> Path:
        path = self.path_for_result(result)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(build_run_log_payload(result), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return path
