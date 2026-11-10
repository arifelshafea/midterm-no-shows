
"""Tests for model comparison, tuning, and threshold selection."""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from noshows.data import make_appointments
from noshows.pipeline import RANDOM_STATE, build_pipeline, split_data
from noshows.train import (
    SCORING,
    get_candidate_models,
    select_call_threshold,
)


# Verify the required baseline and three candidate classifiers exist.
def test_candidate_models():
    models = get_candidate_models()

    assert len(models) >= 4
    assert "dummy" in models
    assert "logistic_regression" in models
    assert "logistic_regression_balanced" in models
    assert "random_forest" in models


# Verify all instructor-required evaluation metrics are configured.
@pytest.mark.parametrize(
    "metric",
    ["precision", "recall", "f1", "roc_auc", "accuracy"],
)
def test_required_scoring_metrics(metric):
    assert metric in SCORING


# Verify that the model's threshold is selected in the allowed range.
def test_out_of_fold_threshold_selection():
    frame = make_appointments(n=200, seed=RANDOM_STATE)
    X_train, _, y_train, _ = split_data(frame)

    pipeline = build_pipeline(
        LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
        )
    )

    threshold, cost = select_call_threshold(
        X_train,
        y_train,
        pipeline,
    )

    assert 0.05 <= threshold <= 0.95
    assert np.isfinite(cost)
    assert cost >= 0
