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


# Verify the final evaluation calculates classification metrics and costs.
def test_final_evaluation_metrics():
    from noshows.train import evaluate_test_set

    class FakePipeline:
        """Return controlled probabilities for a predictable test."""

        def predict_proba(self, X):
            return np.column_stack(
                [1 - np.array([0.9, 0.1, 0.8, 0.2]), [0.9, 0.1, 0.8, 0.2]]
            )

    actual = np.array([1, 0, 1, 0])
    X_test = np.zeros((4, 1))

    results = evaluate_test_set(
        FakePipeline(),
        X_test,
        actual,
        threshold=0.5,
    )

    assert results["accuracy"] == pytest.approx(1.0)
    assert results["precision"] == pytest.approx(1.0)
    assert results["recall"] == pytest.approx(1.0)
    assert results["f1"] == pytest.approx(1.0)
    assert results["roc_auc"] == pytest.approx(1.0)

    assert results["costs"] == {
        "call_nobody": 70,
        "call_everybody": 32,
        "model": 16,
    }

    assert results["beats_both_baselines"] is True


# Verify the saved model and metadata files exist.
def test_model_artifacts_exist():
    from noshows.train import MODEL_DIR

    assert (MODEL_DIR / "no_show.joblib").is_file()
    assert (MODEL_DIR / "model_info.json").is_file()


# Verify the report files were generated.
def test_report_artifacts_exist():
    from noshows.train import REPORT_DIR

    assert (REPORT_DIR / "cv_results.csv").is_file()
    assert (REPORT_DIR / "metrics.json").is_file()


# Verify the saved model metadata contains required information.
def test_model_metadata():
    import json

    import sklearn

    from noshows.train import MODEL_DIR

    with (MODEL_DIR / "model_info.json").open(encoding="utf-8") as file:
        info = json.load(file)

    assert info["scikit_learn_version"] == sklearn.__version__
    assert info["seed"] == RANDOM_STATE
    assert 0.05 <= info["threshold"] <= 0.95
    assert info["features"]
    assert info["best_settings"]
    assert info["test_results"]
