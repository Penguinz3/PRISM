import json
from pathlib import Path

from prism.cli import main


def test_cli_processes_manual_run_example(tmp_path, capsys) -> None:
    example = Path(__file__).resolve().parents[1] / "examples" / "manual_run.json"

    exit_code = main(["--input", str(example), "--output-dir", str(tmp_path)])
    output = capsys.readouterr().out
    payload = json.loads(output)

    assert exit_code == 0
    assert payload["memory_conflicts"]
    assert payload["advisor_diagnostic"]["metadata"]["recommended_action"] == "reject_or_do_not_store"
    assert list(tmp_path.rglob("*.json"))


def test_cli_processes_auto_claim_extraction_example(tmp_path, capsys) -> None:
    example = Path(__file__).resolve().parents[1] / "examples" / "claim_extraction_run.json"

    exit_code = main(["--input", str(example), "--output-dir", str(tmp_path)])
    output = capsys.readouterr().out
    payload = json.loads(output)

    assert exit_code == 0
    assert payload["config"]["claim_extraction"] == "deterministic_stub"
    assert payload["extracted_claims"]
    assert payload["memory_conflicts"]
    assert list(tmp_path.rglob("*.json"))


def test_cli_processes_mock_model_run_example(tmp_path, capsys) -> None:
    example = Path(__file__).resolve().parents[1] / "examples" / "mock_model_run.json"

    exit_code = main(["--input", str(example), "--output-dir", str(tmp_path)])
    output = capsys.readouterr().out
    payload = json.loads(output)

    assert exit_code == 0
    assert payload["config"]["mode"] == "model_backend"
    assert payload["model_info"]["backend_name"] == "mock"
    assert payload["primary_answer"]["text"] == "PRISM uses semantic entropy. PRISM has KG memory."
    assert payload["sample_set"]["samples"]
    assert payload["primary_answer"]["token_confidences"]
    assert list(tmp_path.rglob("*.json"))
