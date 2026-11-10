"""Tests for business costs and threshold optimization."""

import numpy as np
import pytest

from noshows.evaluation import (
    THRESHOLDS,
    calculate_cost,
    choose_threshold,
    compare_costs,
)


# Check the cost calculation against an example calculated by hand.
def test_cost_hand_calculation():
    # Four appointments:
    # Call patients 1 and 3: 2 × $8 = $16.
    # Patient 2 misses the appointment without a call: $35.
    # Total cost = $51.
    actual = [1, 1, 0, 0]
    calls = [True, False, True, False]

    assert calculate_cost(actual, calls) == 51


# Verify the total cost when nobody is called.
def test_call_nobody_cost():
    actual = [1, 0, 1, 0]
    calls = [False, False, False, False]

    assert calculate_cost(actual, calls) == 70


# Verify the total cost when everybody is called.
def test_call_everybody_cost():
    actual = [1, 0, 1, 0]
    calls = [True, True, True, True]

    assert calculate_cost(actual, calls) == 32


# Test situations where calling nobody or everybody is cheapest.
@pytest.mark.parametrize(
    ("actual", "expected_best"),
    [
        ([0, 0, 0, 0], "call_nobody"),
        ([1, 1, 1, 1], "call_everybody"),
    ],
)
def test_cheapest_baseline_strategy(actual, expected_best):
    probabilities = [0.5] * len(actual)
    costs = compare_costs(actual, probabilities, threshold=0.5)

    assert costs[expected_best] == min(costs.values())


# Verify that the allowed threshold range matches the instructions.
def test_threshold_range():
    assert THRESHOLDS[0] == pytest.approx(0.05)
    assert THRESHOLDS[-1] == pytest.approx(0.95)


# When every patient attends, selecting no calls is cheapest.
def test_threshold_when_calls_are_not_worth_it():
    actual = [0, 0, 0, 0]
    probabilities = [0.1, 0.2, 0.3, 0.4]

    threshold, cost = choose_threshold(actual, probabilities)

    assert threshold > 0.4
    assert cost == 0


# When every patient misses their appointment, calling everyone is cheapest.
def test_threshold_when_misses_are_very_costly():
    actual = [1, 1, 1, 1]
    probabilities = [0.2, 0.3, 0.4, 0.5]

    threshold, cost = choose_threshold(actual, probabilities)

    assert threshold <= 0.2
    assert cost == 32


# Verify all three business strategies return the expected costs.
def test_compare_costs():
    actual = [1, 0, 1, 0]
    probabilities = [0.8, 0.1, 0.2, 0.9]

    costs = compare_costs(actual, probabilities, threshold=0.5)

    assert costs == {
        "call_nobody": 70,
        "call_everybody": 32,
        "model": 51,
    }


# Reject inputs when the labels and decisions have different lengths.
def test_cost_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="equal shapes"):
        calculate_cost([1, 0], [True])


# Reject inputs when labels and probabilities have different lengths.
def test_threshold_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="equal shapes"):
        choose_threshold([1, 0], [0.5])


# Verify that costs are always nonnegative.
def test_cost_is_nonnegative():
    actual = np.array([1, 0, 1])
    calls = np.array([True, False, False])

    assert calculate_cost(actual, calls) >= 0
