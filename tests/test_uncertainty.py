import math

import pytest

from prism import GeneratedAnswer, RiskLevel, SampleSet, TokenConfidence, UncertaintyScores
from prism.uncertainty import (
    average_pairwise_consistency,
    build_uncertainty_scores,
    compute_reliability_score,
    compute_risk_score,
    exact_match_consistency,
    logprob_confidence,
    mean_logprob,
    normalized_semantic_entropy,
    semantic_entropy_from_clusters,
    semantic_entropy_from_samples,
    simple_jaccard_similarity,
)


def test_mean_logprob_from_token_confidences() -> None:
    tokens = [
        TokenConfidence(token="A", logprob=-0.1),
        TokenConfidence(token="B", logprob=-0.3),
    ]

    assert mean_logprob(tokens) == pytest.approx(-0.2)


def test_logprob_confidence_normalization() -> None:
    tokens = [
        TokenConfidence(token="A", logprob=-0.1),
        TokenConfidence(token="B", logprob=-0.3),
    ]

    assert logprob_confidence(tokens) == pytest.approx(math.exp(-0.2))


def test_empty_token_list_behavior() -> None:
    assert mean_logprob([]) is None
    assert logprob_confidence([]) is None


def test_exact_match_consistency() -> None:
    assert exact_match_consistency(["Answer!", " answer ", "Different"]) == pytest.approx(2 / 3)
    assert exact_match_consistency([]) is None


def test_jaccard_similarity() -> None:
    score = simple_jaccard_similarity("PRISM uses KG memory", "PRISM uses memory graph")

    assert score == pytest.approx(3 / 5)
    assert simple_jaccard_similarity("", "") == 1.0
    assert simple_jaccard_similarity("PRISM", "") == 0.0


def test_average_pairwise_consistency() -> None:
    samples = ["alpha beta", "alpha beta", "alpha gamma"]

    assert average_pairwise_consistency(samples) == pytest.approx(5 / 9)
    assert average_pairwise_consistency([]) is None
    assert average_pairwise_consistency(["one"]) == 1.0


def test_average_pairwise_consistency_accepts_sample_set() -> None:
    primary = GeneratedAnswer(prompt="Q", text="alpha beta")
    sample = GeneratedAnswer(prompt="Q", text="alpha gamma")
    sample_set = SampleSet(prompt="Q", primary_answer=primary, samples=[sample])

    assert average_pairwise_consistency(sample_set) == pytest.approx(1 / 3)


def test_semantic_entropy_from_one_cluster_is_zero() -> None:
    assert semantic_entropy_from_clusters([4]) == 0.0
    assert normalized_semantic_entropy([4]) == 0.0


def test_semantic_entropy_increases_with_balanced_clusters() -> None:
    skewed = semantic_entropy_from_clusters([3, 1])
    balanced = semantic_entropy_from_clusters([2, 2])

    assert balanced > skewed


def test_normalized_semantic_entropy_in_range() -> None:
    score = normalized_semantic_entropy([2, 1, 1])

    assert 0.0 <= score <= 1.0
    assert normalized_semantic_entropy([1, 1, 1]) == pytest.approx(1.0)


def test_full_semantic_entropy_placeholder_raises() -> None:
    with pytest.raises(NotImplementedError, match="semantic clustering"):
        semantic_entropy_from_samples(["A", "B"])


def test_risk_score_clamping() -> None:
    assert compute_risk_score(
        logprob_confidence=1.0,
        self_consistency=1.0,
        kg_support=1.0,
        normalized_semantic_entropy=0.0,
    ) == 0.0
    assert compute_risk_score(
        logprob_confidence=0.0,
        self_consistency=0.0,
        kg_support=0.0,
        normalized_semantic_entropy=1.0,
    ) == 1.0


def test_reliability_improves_with_higher_kg_support_and_self_consistency() -> None:
    low = compute_reliability_score(
        logprob_confidence=0.5,
        self_consistency=0.2,
        kg_support=0.2,
        normalized_semantic_entropy=0.2,
    )
    high = compute_reliability_score(
        logprob_confidence=0.5,
        self_consistency=0.8,
        kg_support=0.8,
        normalized_semantic_entropy=0.2,
    )

    assert high is not None
    assert low is not None
    assert high > low


def test_reliability_decreases_with_higher_entropy() -> None:
    low_entropy = compute_reliability_score(
        logprob_confidence=0.7,
        self_consistency=0.7,
        kg_support=0.7,
        normalized_semantic_entropy=0.1,
    )
    high_entropy = compute_reliability_score(
        logprob_confidence=0.7,
        self_consistency=0.7,
        kg_support=0.7,
        normalized_semantic_entropy=0.9,
    )

    assert low_entropy is not None
    assert high_entropy is not None
    assert low_entropy > high_entropy


def test_build_uncertainty_scores_helper() -> None:
    tokens = [
        TokenConfidence(token="A", logprob=-0.1),
        TokenConfidence(token="B", logprob=-0.3),
    ]

    scores = build_uncertainty_scores(
        token_confidences=tokens,
        samples=["alpha beta", "alpha gamma"],
        cluster_sizes=[1, 1],
        kg_support=0.8,
        details={"source": "unit-test"},
    )

    assert isinstance(scores, UncertaintyScores)
    assert scores.mean_logprob == pytest.approx(-0.2)
    assert scores.mean_token_probability == pytest.approx(math.exp(-0.2))
    assert scores.self_consistency_disagreement == pytest.approx(2 / 3)
    assert scores.semantic_entropy == pytest.approx(math.log(2))
    assert scores.semantic_entropy_normalized == pytest.approx(1.0)
    assert scores.combined_risk_score is not None
    assert scores.risk_level in {
        RiskLevel.LOW,
        RiskLevel.MEDIUM,
        RiskLevel.HIGH,
        RiskLevel.CRITICAL,
    }
    assert scores.details["kg_support"] == 0.8
    assert scores.details["source"] == "unit-test"
