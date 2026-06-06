"""Minimal command line entrypoint for manual or model-backed PRISM runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from prism.memory import MemoryGraphStore
from prism.pipeline import run_prism_analysis
from prism.schemas import ClaimTriple, UncertaintyScores


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _claims_from_payload(items) -> list[ClaimTriple]:
    return [ClaimTriple.from_dict(item) for item in (items or [])]


def _uncertainty_from_payload(payload: dict[str, Any]):
    data = payload.get("uncertainty_scores")
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run PRISM analysis over manual or model inputs.")
    parser.add_argument("--input", help="Path to a PRISM manual run JSON file.")
    parser.add_argument("--prompt", help="Prompt text for direct manual input.")
    parser.add_argument("--answer", help="Answer text for direct manual input.")
    parser.add_argument("--claims", help="JSON list of manual claim objects.")
    parser.add_argument("--memory-claims", help="JSON list of memory claim objects.")
    parser.add_argument("--model-backend", help="Model backend name for answer-omitted runs.")
    parser.add_argument("--model-name", help="Model name for the selected backend.")
    parser.add_argument("--mock-answer", help="Deterministic mock answer for --model-backend mock.")
    parser.add_argument("--sample-count", type=int, default=0, help="Number of backend samples to request.")
    parser.add_argument("--output-dir", default="logs/runs", help="Run log root directory.")
    parser.add_argument("--no-log", action="store_true", help="Run without writing a JSON log.")
    return parser


def _payload_from_args(args: argparse.Namespace) -> dict[str, Any]:
    if args.input:
        return _load_json(args.input)
    if not args.prompt or (not args.answer and not args.model_backend):
        raise ValueError("provide --input, both --prompt and --answer, or --prompt with --model-backend")
    return {
        "prompt": args.prompt,
        "answer": args.answer,
        "claims": json.loads(args.claims or "[]"),
        "memory_claims": json.loads(args.memory_claims or "[]"),
        "model_backend": args.model_backend,
        "model_name": args.model_name,
        "mock_answer": args.mock_answer,
        "sample_count": args.sample_count,
    }


def _model_backend_config(payload: dict[str, Any]) -> dict[str, Any] | None:
    if not payload.get("model_backend"):
        return None
    config = dict(payload.get("model_backend_config", {}))
    for key in (
        "base_url",
        "endpoint",
        "include_token_confidences",
        "mock_answer",
        "mock_samples",
        "model_name",
        "pipeline_kwargs",
        "samples",
        "task",
        "timeout",
        "token_confidences",
    ):
        if payload.get(key) is not None:
            config[key] = payload[key]
    return config


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        payload = _payload_from_args(args)
        memory_store = MemoryGraphStore(_claims_from_payload(payload.get("memory_claims")))
        result = run_prism_analysis(
            prompt=payload["prompt"],
            answer=payload.get("answer"),
            claims=_claims_from_payload(payload.get("claims")),
            memory_store=memory_store,
            uncertainty_scores=_uncertainty_from_payload(payload),
            write_log=not args.no_log,
            output_dir=args.output_dir,
            auto_extract_claims=bool(payload.get("auto_extract_claims", False)),
            model_backend=payload.get("model_backend"),
            model_backend_config=_model_backend_config(payload),
            sample_count=int(payload.get("sample_count", 0) or 0),
            generation_config=payload.get("generation_config", {}),
            config={"cli_input": str(args.input) if args.input else "direct_args"},
        )
    except Exception as exc:  # pragma: no cover - exercised through CLI behavior
        parser.exit(2, f"prism.cli error: {exc}\n")

    sys.stdout.write(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
