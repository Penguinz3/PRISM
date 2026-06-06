import builtins
import json
import urllib.error

import pytest

from prism.config import get_model_backend
from prism.models import (
    HuggingFaceModelBackend,
    MockModelBackend,
    ModelBackend,
    ModelBackendError,
    OllamaBackend,
    OpenAICompatibleLocalBackend,
)
from prism.schemas import GeneratedAnswer, SampleSet


class FakeResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def test_base_interface_shape() -> None:
    assert hasattr(ModelBackend, "generate")
    assert hasattr(ModelBackend, "sample")
    assert callable(MockModelBackend().generate)
    assert callable(MockModelBackend().sample)


def test_mock_model_backend_generate() -> None:
    backend = MockModelBackend(
        answer="PRISM uses semantic entropy.",
        model_name="mock-test",
        include_token_confidences=True,
    )

    answer = backend.generate("What does PRISM use?", temperature=0.0)

    assert isinstance(answer, GeneratedAnswer)
    assert answer.text == "PRISM uses semantic entropy."
    assert answer.model_name == "mock-test"
    assert answer.metadata["backend_name"] == "mock"
    assert answer.sampling_parameters["temperature"] == 0.0
    assert answer.token_confidences


def test_mock_model_backend_sample() -> None:
    backend = MockModelBackend(
        answer="Primary answer.",
        samples=["Sample one.", "Sample two."],
    )

    sample_set = backend.sample("Prompt?", n=3)

    assert isinstance(sample_set, SampleSet)
    assert sample_set.primary_answer.text == "Primary answer."
    assert [sample.text for sample in sample_set.samples] == [
        "Sample one.",
        "Sample two.",
        "Sample one.",
    ]


def test_backend_factory_returns_mock() -> None:
    backend = get_model_backend("mock", mock_answer="Factory answer.")

    assert isinstance(backend, MockModelBackend)
    assert backend.generate("Prompt").text == "Factory answer."


def test_backend_factory_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError, match="unknown model backend"):
        get_model_backend("not-a-backend")


def test_openai_compatible_backend_handles_mocked_success(monkeypatch) -> None:
    def fake_urlopen(request, timeout):
        sent = json.loads(request.data.decode("utf-8"))
        assert request.full_url == "http://local.test/v1/chat/completions"
        assert timeout == 4
        assert sent["model"] == "local-model"
        assert sent["messages"][0]["content"] == "Prompt?"
        return FakeResponse(
            {
                "id": "cmpl-test",
                "choices": [{"message": {"content": "PRISM uses KG memory."}}],
            }
        )

    monkeypatch.setattr("prism.models.openai_compatible.urlrequest.urlopen", fake_urlopen)
    backend = OpenAICompatibleLocalBackend(
        base_url="http://local.test/v1",
        model_name="local-model",
        timeout=4,
    )

    answer = backend.generate("Prompt?", temperature=0.1)

    assert answer.text == "PRISM uses KG memory."
    assert answer.metadata["backend_name"] == "openai_compatible"
    assert answer.metadata["response_id"] == "cmpl-test"
    assert answer.sampling_parameters["temperature"] == 0.1


def test_ollama_backend_handles_mocked_success(monkeypatch) -> None:
    def fake_urlopen(request, timeout):
        sent = json.loads(request.data.decode("utf-8"))
        assert request.full_url == "http://ollama.test/api/generate"
        assert timeout == 5
        assert sent["model"] == "llama-test"
        assert sent["stream"] is False
        return FakeResponse({"response": "PRISM has KG memory.", "done": True})

    monkeypatch.setattr("prism.models.ollama.urlrequest.urlopen", fake_urlopen)
    backend = OllamaBackend(
        base_url="http://ollama.test",
        model_name="llama-test",
        timeout=5,
    )

    answer = backend.generate("Prompt?", temperature=0.2)

    assert answer.text == "PRISM has KG memory."
    assert answer.metadata["backend_name"] == "ollama"
    assert answer.metadata["done"] is True


def test_ollama_unavailable_error_is_clear(monkeypatch) -> None:
    def fake_urlopen(request, timeout):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr("prism.models.ollama.urlrequest.urlopen", fake_urlopen)
    backend = OllamaBackend(base_url="http://ollama.test")

    with pytest.raises(ModelBackendError, match="Ollama backend unavailable"):
        backend.generate("Prompt?")


def test_huggingface_imports_are_lazy_and_missing_dependency_is_clear(monkeypatch) -> None:
    backend = HuggingFaceModelBackend(model_name="tiny-test-model")
    assert backend.model_name == "tiny-test-model"

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "transformers":
            raise ImportError("blocked for test")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match="requires optional dependencies"):
        backend.generate("Prompt?")
