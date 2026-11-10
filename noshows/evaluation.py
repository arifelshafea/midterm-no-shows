
"""Business cost calculations and decision-threshold selection."""

import numpy as np

# Business costs from the instructor's scenario.
CALL_COST = 8
MISSED_APPOINTMENT_COST = 35

# Candidate thresholds required by the instructor.
THRESHOLDS = np.round(np.arange(0.05, 0.951, 0.01), 2)


# Calculate the total cost of calling selected patients.
def calculate_cost(y_true, call_decisions):
    """Return call costs plus the cost of uncalled no-shows."""
    actual = np.asarray(y_true, dtype=int)
    calls = np.asarray(call_decisions, dtype=bool)

    if actual.shape != calls.shape:
        raise ValueError("Actual labels and call decisions must have equal shapes.")

    call_total = CALL_COST * np.count_nonzero(calls)
    missed_total = MISSED_APPOINTMENT_COST * np.count_nonzero(
        (actual == 1) & ~calls
    )

    return int(call_total + missed_total)


# Choose the threshold that minimizes cost on training predictions.
def choose_threshold(y_true, probabilities):
    """Find the lowest-cost threshold using out-of-fold probabilities."""
    actual = np.asarray(y_true)
    scores = np.asarray(probabilities)

    if actual.shape != scores.shape:
        raise ValueError("Labels and probabilities must have equal shapes.")

    best_threshold = None
    best_cost = float("inf")

    for threshold in THRESHOLDS:
        calls = scores >= threshold
        cost = calculate_cost(actual, calls)

        if cost < best_cost:
            best_cost = cost
            best_threshold = float(threshold)

    return best_threshold, best_cost


# Compare the three calling strategies on the same data.
def compare_costs(y_true, probabilities, threshold):
    """Return costs for calling nobody, everybody, or using the model."""
    actual = np.asarray(y_true)
    scores = np.asarray(probabilities)

    no_calls = np.zeros(len(actual), dtype=bool)
    all_calls = np.ones(len(actual), dtype=bool)
    model_calls = scores >= threshold

    return {
        "call_nobody": calculate_cost(actual, no_calls),
        "call_everybody": calculate_cost(actual, all_calls),
        "model": calculate_cost(actual, model_calls),
    }
