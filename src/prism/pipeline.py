from dataclasses import dataclass

from prism.model import LLMWrapper
from prism.markers.base import MarkerInput, MarkerResult
from prism.markers.mean_logprob import MeanLogProbMarker


@dataclass
class AnalysisResult:
    prompt: str
    primary_text: str
    sample_texts: list[str]
    marker_results: list[MarkerResult]


class PRISMPipeline:
    def __init__(self, model: LLMWrapper, markers: list):
        self.model = model
        self.markers = markers
        # Inject the tokenizer into any MeanLogProbMarker that doesn't have one yet
        for marker in self.markers:
            if isinstance(marker, MeanLogProbMarker) and marker.tokenizer is None:
                marker.tokenizer = self.model.tokenizer

    def analyze(self, prompt: str, num_samples: int = 5) -> AnalysisResult:
        primary_outputs = self.model.generate(
            prompt, temperature=0, max_new_tokens=200, num_return_sequences=1
        )
        primary = primary_outputs[0]

        sample_outputs = self.model.generate(
            prompt, temperature=0.7, max_new_tokens=200, num_return_sequences=num_samples
        )

        marker_input = MarkerInput(
            prompt=prompt,
            primary_response=primary,
            sampled_responses=sample_outputs,
        )

        marker_results = [marker.compute(marker_input) for marker in self.markers]

        return AnalysisResult(
            prompt=prompt,
            primary_text=primary.text,
            sample_texts=[r.text for r in sample_outputs],
            marker_results=marker_results,
        )
