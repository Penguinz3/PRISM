from dataclasses import dataclass, field
from typing import Protocol

from prism.model import GenerationOutput


@dataclass
class MarkerInput:
    prompt: str
    primary_response: GenerationOutput
    sampled_responses: list[GenerationOutput]


@dataclass
class MarkerResult:
    name: str
    score: float
    threshold: float
    triggered: bool
    direction: str
    details: dict = field(default_factory=dict)
    explanation: str = ""


class BaseMarker(Protocol):
    name: str

    def compute(self, input: MarkerInput) -> MarkerResult: ...
