"""Semantic entropy interfaces.

Full semantic clustering is intentionally deferred. This module only computes
Shannon entropy when cluster assignments have already been provided.
"""

from __future__ import annotations

import math
from typing import Iterable


def _clean_cluster_sizes(cluster_sizes: Iterable[int]) -> tuple[int, ...]:
    sizes = tuple(int(size) for size in cluster_sizes)
    if any(size < 0 for size in sizes):
        raise ValueError("cluster sizes must be non-negative")
    return tuple(size for size in sizes if size > 0)


def semantic_entropy_from_clusters(cluster_sizes: Iterable[int]) -> float:
    """Compute Shannon entropy over cluster proportions.

    If there are no samples or only one non-empty cluster, entropy is `0.0`.
    """

    sizes = _clean_cluster_sizes(cluster_sizes)
    total = sum(sizes)
    if total == 0 or len(sizes) <= 1:
        return 0.0

    entropy = 0.0
    for size in sizes:
        probability = size / total
        entropy -= probability * math.log(probability)
    return entropy


def normalized_semantic_entropy(cluster_sizes: Iterable[int]) -> float:
    """Return semantic entropy normalized by `log(k)` for `k` clusters."""

    sizes = _clean_cluster_sizes(cluster_sizes)
    if len(sizes) <= 1:
        return 0.0
    max_entropy = math.log(len(sizes))
    if max_entropy == 0.0:
        return 0.0
    score = semantic_entropy_from_clusters(sizes) / max_entropy
    return max(0.0, min(1.0, score))


def semantic_entropy_from_samples(*_args, **_kwargs) -> float:
    """Placeholder for future embedding/NLI semantic clustering."""

    raise NotImplementedError(
        "Full semantic entropy requires semantic clustering via embeddings or NLI; "
        "provide cluster sizes to semantic_entropy_from_clusters instead."
    )
