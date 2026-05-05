import math

import numpy as np
from sentence_transformers import CrossEncoder

from prism import config
from .base import BaseMarker, MarkerInput, MarkerResult

# Label order for cross-encoder/nli-deberta-v3-small: contradiction=0, entailment=1, neutral=2
_ENTAILMENT_IDX = 1


class SemanticEntropyMarker:
    name = "semantic_entropy"

    def __init__(self, threshold: float, nli_model_name: str = config.NLI_MODEL_NAME):
        self.threshold = threshold
        self._nli = CrossEncoder(nli_model_name)

    def _bidirectional_entailment(self, a: str, b: str) -> bool:
        scores = self._nli.predict([(a, b), (b, a)])
        return (
            int(np.argmax(scores[0])) == _ENTAILMENT_IDX
            and int(np.argmax(scores[1])) == _ENTAILMENT_IDX
        )

    def _cluster(self, responses: list[str]) -> list[list[int]]:
        clusters: list[list[int]] = []
        for i, response in enumerate(responses):
            placed = False
            for cluster in clusters:
                rep = responses[cluster[0]]
                if self._bidirectional_entailment(rep, response):
                    cluster.append(i)
                    placed = True
                    break
            if not placed:
                clusters.append([i])
        return clusters

    def compute(self, input: MarkerInput) -> MarkerResult:
        all_responses = [input.primary_response.text] + [
            r.text for r in input.sampled_responses
        ]
        clusters = self._cluster(all_responses)

        n = len(all_responses)
        entropy = 0.0
        for cluster in clusters:
            p = len(cluster) / n
            entropy -= p * math.log(p)

        return MarkerResult(
            name=self.name,
            score=entropy,
            threshold=self.threshold,
            triggered=entropy > self.threshold,
            direction="higher_is_worse",
            details={
                "clusters": clusters,
                "cluster_count": len(clusters),
            },
        )
