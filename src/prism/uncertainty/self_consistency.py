"""Lightweight text-only self-consistency baselines.

These functions are placeholders for future embedding/NLI methods. They are
deterministic, dependency-free, and useful as sanity-check baselines only.
"""

from __future__ import annotations

import string
from itertools import combinations
from typing import Iterable

from prism.schemas import GeneratedAnswer, SampleSet


_PUNCTUATION_TABLE = str.maketrans({char: " " for char in string.punctuation})


def _normalize_text(text: str) -> str:
    return " ".join(text.lower().translate(_PUNCTUATION_TABLE).split())


def _tokens(text: str) -> set[str]:
    normalized = _normalize_text(text)
    if not normalized:
        return set()
    return set(normalized.split())


def _coerce_texts(samples: SampleSet | Iterable[GeneratedAnswer | str]) -> tuple[str, ...]:
    if isinstance(samples, SampleSet):
        return tuple(answer.text for answer in samples.all_answers)

    texts: list[str] = []
    for sample in samples:
        if isinstance(sample, GeneratedAnswer):
            texts.append(sample.text)
        elif isinstance(sample, str):
            texts.append(sample)
        else:
            raise TypeError("samples must contain GeneratedAnswer objects or strings")
    return tuple(texts)


def simple_jaccard_similarity(a: str, b: str) -> float:
    """Return token-set Jaccard similarity in `[0, 1]`."""

    a_tokens = _tokens(a)
    b_tokens = _tokens(b)
    if not a_tokens and not b_tokens:
        return 1.0
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)


def exact_match_consistency(samples: SampleSet | Iterable[GeneratedAnswer | str]) -> float | None:
    """Return the fraction of texts matching the most common normalized answer.

    Empty inputs return `None`. A single sample is perfectly self-consistent.
    """

    texts = _coerce_texts(samples)
    if not texts:
        return None

    counts: dict[str, int] = {}
    for text in texts:
        normalized = _normalize_text(text)
        counts[normalized] = counts.get(normalized, 0) + 1
    return max(counts.values()) / len(texts)


def average_pairwise_consistency(samples: SampleSet | Iterable[GeneratedAnswer | str]) -> float | None:
    """Return average pairwise Jaccard similarity across answer texts."""

    texts = _coerce_texts(samples)
    if not texts:
        return None
    if len(texts) == 1:
        return 1.0

    similarities = [simple_jaccard_similarity(a, b) for a, b in combinations(texts, 2)]
    return sum(similarities) / len(similarities)
