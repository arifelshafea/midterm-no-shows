"""Tests for model comparison, tuning, and threshold selection."""

import json

import numpy as np
import pandas as pd
import pytest
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

import noshows.train as training
from noshows.data import make_appointments
from noshows.pipeline import FEATURES, RANDOM_STATE, build_pipeline, split_data
from noshows.train import (
    MODEL_DIR,
    REPORT_DIR,
    SCORING,
    evaluate_test_set,
    get_candidate_models,
    select_call_threshold,
    tune_model,
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


# Verify that the model threshold is selected in the allowed range.
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


# Verify final evaluation calculates classification metrics and costs.
def test_final_evaluation_metrics():
    class FakePipeline:
        """Return controlled probabilities for predictable testing."""

        def predict_proba(self, X):
            probabilities = np.array([0.9, 0.1, 0.8, 0.2])
            return np.column_stack([1 - probabilities, probabilities])

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


# Verify the saved fitted pipeline and metadata files exist.
def test_model_artifacts_exist():
    assert (MODEL_DIR / "no_show.joblib").is_file()
    assert (MODEL_DIR / "model_info.json").is_file()


# Verify the required report files exist.
def test_report_artifacts_exist():
    assert (REPORT_DIR / "cv_results.csv").is_file()
    assert (REPORT_DIR / "metrics.json").is_file()


# Verify the saved model metadata contains required information.
def test_model_metadata():
    with (MODEL_DIR / "model_info.json").open(encoding="utf-8") as file:
        info = json.load(file)

    assert info["scikit_learn_version"] == sklearn.__version__
    assert info["seed"] == RANDOM_STATE
    assert 0.05 <= info["threshold"] <= 0.95
    assert info["features"]
    assert info["best_settings"]
    assert info["test_results"]


# Verify GridSearchCV has two parameters and at least six combinations.
def test_grid_search_configuration(monkeypatch):
    captured = {}

    # Inspect tuning configuration without performing an expensive fit.
    def fake_fit(self, X, y):
        captured["params"] = self.param_grid
        captured["scoring"] = self.scoring
        captured["folds"] = self.cv.n_splits
        return self

    monkeypatch.setattr(GridSearchCV, "fit", fake_fit)

    frame = make_appointments(n=100, seed=RANDOM_STATE)
    X_train, _, y_train, _ = split_data(frame)

    tune_model(X_train, y_train)

    parameter_grid = captured["params"]

    combinations = len(parameter_grid["model__C"]) * len(
        parameter_grid["model__fit_intercept"]
    )

    assert len(parameter_grid) >= 2
    assert combinations >= 6
    assert captured["scoring"] == "roc_auc"
    assert captured["folds"] == 5


# Verify threshold selection requests out-of-fold probabilities.
def test_threshold_uses_out_of_fold_predictions(monkeypatch):
    captured = {}

    # Replace predictions with controlled probabilities.
    def fake_cross_val_predict(model, X, y, *, cv, method, n_jobs):
        captured["method"] = method
        captured["folds"] = cv.n_splits

        return np.tile([0.8, 0.2], (len(X), 1))

    monkeypatch.setattr(
        training,
        "cross_val_predict",
        fake_cross_val_predict,
    )

    frame = make_appointments(n=100, seed=RANDOM_STATE)
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

    assert captured["method"] == "predict_proba"
    assert captured["folds"] == 5
    assert 0.05 <= threshold <= 0.95
    assert cost >= 0


# Verify the saved model metadata and evaluation reports.
def test_saved_report_structure():
    with (MODEL_DIR / "model_info.json").open(encoding="utf-8") as file:
        model_info = json.load(file)

    with (REPORT_DIR / "metrics.json").open(encoding="utf-8") as file:
        metrics = json.load(file)

    cv_results = pd.read_csv(REPORT_DIR / "cv_results.csv")

    # Confirm correct features and threshold.
    assert model_info["features"] == FEATURES
    assert model_info["numeric_features"]
    assert model_info["categorical_features"]
    assert 0.05 <= model_info["threshold"] <= 0.95

    # Confirm that the model comparison report includes all four models.
    assert len(cv_results) >= 4
    assert "roc_auc_mean" in cv_results.columns
    assert "roc_auc_std" in cv_results.columns

    # Confirm the required evaluation information is recorded.
    assert "tuning" in metrics
    assert "leakage_comparison" in metrics
    assert "test_results" in metrics

    costs = metrics["test_results"]["costs"]

    assert set(costs) == {
        "call_nobody",
        "call_everybody",
        "model",
    }

    assert metrics["selected_threshold"] == model_info["threshold"]
