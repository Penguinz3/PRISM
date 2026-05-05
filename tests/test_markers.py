import pytest
from prism.model import GenerationOutput
from prism.markers.base import MarkerInput
from prism.markers.self_consistency import SelfConsistencyMarker
from prism.markers.semantic_entropy import SemanticEntropyMarker
from prism import config

_DUMMY_LOGPROBS = [-0.5, -0.3, -0.2]
_DUMMY_IDS = [1, 2, 3]


def _make_output(text: str) -> GenerationOutput:
    return GenerationOutput(text=text, token_ids=_DUMMY_IDS, token_logprobs=_DUMMY_LOGPROBS)


def _make_input(primary_text: str, sample_texts: list[str]) -> MarkerInput:
    return MarkerInput(
        prompt="test prompt",
        primary_response=_make_output(primary_text),
        sampled_responses=[_make_output(t) for t in sample_texts],
    )


@pytest.fixture(scope="module")
def sc_marker():
    return SelfConsistencyMarker(threshold=config.MARKER_THRESHOLDS["self_consistency"])


@pytest.fixture(scope="module")
def se_marker():
    return SemanticEntropyMarker(threshold=config.MARKER_THRESHOLDS["semantic_entropy"])


def test_identical_responses_high_consistency_low_entropy(sc_marker, se_marker):
    text = "The capital of France is Paris."
    inp = _make_input(text, [text] * 4)

    sc_result = sc_marker.compute(inp)
    se_result = se_marker.compute(inp)

    assert sc_result.score > 0.95, f"Expected high self-consistency, got {sc_result.score:.3f}"
    assert se_result.score < 0.2, f"Expected low semantic entropy, got {se_result.score:.3f}"
    assert not sc_result.triggered
    assert not se_result.triggered


def test_varied_responses_low_consistency_high_entropy(sc_marker, se_marker):
    primary = "Paris is the capital of France."
    samples = [
        "The Moon orbits the Earth at roughly 384,000 km.",
        "Photosynthesis converts sunlight into chemical energy in plants.",
        "The Great Wall of China was built over many centuries.",
        "Jazz music originated in New Orleans in the early 20th century.",
    ]
    inp = _make_input(primary, samples)

    sc_result = sc_marker.compute(inp)
    se_result = se_marker.compute(inp)

    assert sc_result.score < 0.7, f"Expected low self-consistency, got {sc_result.score:.3f}"
    assert se_result.score > 0.5, f"Expected high semantic entropy, got {se_result.score:.3f}"
    assert se_result.triggered
