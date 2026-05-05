import pytest
from prism.model import LLMWrapper
from prism.pipeline import PRISMPipeline, AnalysisResult
from prism.markers.mean_logprob import MeanLogProbMarker
from prism.markers.self_consistency import SelfConsistencyMarker
from prism.markers.semantic_entropy import SemanticEntropyMarker
from prism import config


@pytest.fixture(scope="module")
def pipeline():
    return PRISMPipeline(
        model=LLMWrapper(config.DEFAULT_MODEL_NAME),
        markers=[
            MeanLogProbMarker(threshold=config.MARKER_THRESHOLDS["mean_logprob"]),
            SelfConsistencyMarker(threshold=config.MARKER_THRESHOLDS["self_consistency"]),
            SemanticEntropyMarker(threshold=config.MARKER_THRESHOLDS["semantic_entropy"]),
        ],
    )


def test_analyze_returns_three_marker_results(pipeline):
    result = pipeline.analyze("What is the capital of France?")

    assert isinstance(result, AnalysisResult)
    assert len(result.marker_results) == 3
    assert isinstance(result.primary_text, str) and len(result.primary_text) > 0
    assert result.prompt == "What is the capital of France?"

    names = {r.name for r in result.marker_results}
    assert names == {"mean_logprob", "self_consistency", "semantic_entropy"}
