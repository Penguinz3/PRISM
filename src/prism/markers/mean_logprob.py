from .base import BaseMarker, MarkerInput, MarkerResult


class MeanLogProbMarker:
    name = "mean_logprob"

    def __init__(self, threshold: float, tokenizer=None):
        self.tokenizer = tokenizer
        self.threshold = threshold

    def compute(self, input: MarkerInput) -> MarkerResult:
        logprobs = input.primary_response.token_logprobs
        score = sum(logprobs) / len(logprobs) if logprobs else 0.0

        if self.tokenizer is not None:
            tokens = [
                self.tokenizer.decode([tid]) for tid in input.primary_response.token_ids
            ]
        else:
            tokens = [str(tid) for tid in input.primary_response.token_ids]

        return MarkerResult(
            name=self.name,
            score=score,
            threshold=self.threshold,
            triggered=score < self.threshold,
            direction="lower_is_worse",
            details={
                "per_token_logprobs": logprobs,
                "tokens": tokens,
            },
        )
