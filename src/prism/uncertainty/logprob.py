"""Logprob-based confidence helpers.

These functions only consume token confidence data that has already been
produced elsewhere. They do not run a model or extract logits.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping

from prism.schemas import TokenConfidence


TokenInput = TokenConfidence | Mapping[str, Any]


def _coerce_token(token: TokenInput) -> TokenConfidence:
    if isinstance(token, TokenConfidence):
        return token
    return TokenConfidence.from_dict(token)


def mean_logprob(token_confidences: Iterable[TokenInput]) -> float | None:
    """Return the arithmetic mean over provided token logprobs.

    Tokens without a `logprob` value are ignored. If the iterable is empty or no
    token has a logprob, the score is unavailable and `None` is returned.
    """

    logprobs = [
        token.logprob
        for token in (_coerce_token(token) for token in token_confidences)
        if token.logprob is not None
    ]
    if not logprobs:
        return None
    return sum(logprobs) / len(logprobs)


def logprob_confidence(token_confidences: Iterable[TokenInput]) -> float | None:
    """Convert mean natural-log probability into a `[0, 1]` confidence.

    For model logprobs, `exp(mean_logprob)` is the geometric mean token
    probability. Positive values are clamped to `1.0` for defensive handling of
    malformed inputs.
    """

    score = mean_logprob(token_confidences)
    if score is None:
        return None
    return max(0.0, min(1.0, math.exp(score)))
