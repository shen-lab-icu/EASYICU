"""Independent finite-distribution checks for a fixed-weight patient bound."""

import itertools
import math

import pytest


def _interval(events, sizes, confidence=0.95):
    from easyicu.research_agent.methods.cluster_proportion import (
        patient_weighted_hoeffding_interval,
    )

    return patient_weighted_hoeffding_interval(
        events, sum(sizes), cluster_count=len(sizes),
        cluster_size_square_sum=sum(size * size for size in sizes),
        confidence_level=confidence,
    )


@pytest.mark.parametrize("events", [0, 100])
def test_a_dominant_patient_cannot_be_counted_as_two_equal_trials(events):
    assert _interval(events, [99, 1]) == (0.0, 100.0)


@pytest.mark.parametrize("confidence", [0.9, 0.95, 0.99])
def test_balanced_bound_matches_the_published_inequality_and_rounds_outward(confidence):
    low, high = _interval(30, [2] * 50, confidence)
    radius = math.sqrt(math.log(2 / (1 - confidence)) / 100)
    assert low <= 100 * (0.3 - radius)
    assert high >= 100 * (0.3 + radius)
    assert abs(low - 100 * (0.3 - radius)) < 1e-6
    assert abs(high - 100 * (0.3 + radius)) < 1e-6


@pytest.mark.parametrize(
    "sizes,probabilities",
    [
        ([99] + [1] * 7, [0.55] + [0.0] * 7),
        ([80, 10, 10], [0.7, 0.1, 0.3]),
        ([2] * 12, [0.03] * 12),
        ([2] * 12, [0.97] * 12),
        ([1, 2, 3, 4, 5, 6], [0.0, 0.05, 0.2, 0.6, 0.95, 1.0]),
    ],
)
def test_exact_coverage_with_perfect_within_patient_dependence(sizes, probabilities):
    target = 100 * sum(m * p for m, p in zip(sizes, probabilities)) / sum(sizes)
    coverage = 0.0
    for states in itertools.product((0, 1), repeat=len(sizes)):
        probability = math.prod(p if y else 1 - p for y, p in zip(states, probabilities))
        if not probability:
            continue
        low, high = _interval(sum(m * y for m, y in zip(sizes, states)), sizes)
        if low <= target <= high:
            coverage += probability
    assert coverage >= 0.95 - 1e-12


def test_exact_coverage_also_allows_independent_stays_within_each_patient():
    sizes, probabilities = [2, 3, 4, 1], [0.1, 0.4, 0.7, 0.9]
    target = 100 * sum(m * p for m, p in zip(sizes, probabilities)) / sum(sizes)
    coverage = 0.0
    for events in itertools.product(*(range(m + 1) for m in sizes)):
        probability = math.prod(
            math.comb(m, k) * p**k * (1 - p)**(m - k)
            for m, k, p in zip(sizes, events, probabilities)
        )
        low, high = _interval(sum(events), sizes)
        if low <= target <= high:
            coverage += probability
    assert coverage >= 0.95 - 1e-12


@pytest.mark.parametrize(
    "updates",
    [
        {"events": -1}, {"events": 101}, {"denominator": 0},
        {"cluster_count": 1}, {"cluster_count": 101},
        {"cluster_size_square_sum": 100}, {"cluster_size_square_sum": 10000},
        {"cluster_size_square_sum": 5001}, {"cluster_size_square_sum": True},
        {"confidence_level": float("nan")}, {"confidence_level": 1.0},
    ],
)
def test_impossible_weight_receipts_or_confidence_are_rejected(updates):
    from easyicu.research_agent.methods.cluster_proportion import (
        patient_weighted_hoeffding_interval,
    )

    kwargs = dict(events=0, denominator=100, cluster_count=2,
                  cluster_size_square_sum=5000, confidence_level=0.95)
    kwargs.update(updates)
    with pytest.raises(ValueError):
        patient_weighted_hoeffding_interval(**kwargs)
