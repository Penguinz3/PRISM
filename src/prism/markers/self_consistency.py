from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from .base import BaseMarker, MarkerInput, MarkerResult

_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class SelfConsistencyMarker:
    name = "self_consistency"

    def __init__(self, threshold: float):
        self.threshold = threshold
        self._embedder = SentenceTransformer(_EMBED_MODEL)

    def compute(self, input: MarkerInput) -> MarkerResult:
        primary_text = input.primary_response.text
        sample_texts = [r.text for r in input.sampled_responses]

        all_texts = [primary_text] + sample_texts
        embeddings = self._embedder.encode(all_texts, convert_to_numpy=True)

        primary_emb = embeddings[0:1]
        sample_embs = embeddings[1:]

        similarities = cosine_similarity(primary_emb, sample_embs)[0].tolist()
        score = sum(similarities) / len(similarities) if similarities else 0.0

        return MarkerResult(
            name=self.name,
            score=score,
            threshold=self.threshold,
            triggered=score < self.threshold,
            direction="lower_is_worse",
            details={
                "sample_similarities": similarities,
                "samples": sample_texts,
            },
        )
