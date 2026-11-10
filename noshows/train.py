"""Train, evaluate, and save the Maple Physio no-show model."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
)

from noshows.data import make_appointments
from noshows.evaluation import choose_threshold, compare_costs
from noshows.pipeline import (
    CATEGORICAL_FEATURES,
    FEATURES,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    TEST_SIZE,
    build_pipeline,
    split_data,
)

# Locate the project root so paths work from different directories.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

# Required evaluation metrics.
SCORING = {
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
    "accuracy": "accuracy",
}

# Five-fold stratified cross-validation.
CV = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE,
)


# Define the required baseline and candidate classifiers.
def get_candidate_models():
    """Return the baseline and three candidate models."""
    return {
        "dummy": DummyClassifier(strategy="prior"),
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
        ),
        "logistic_regression_balanced": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


# Compare models using the training set only.
def compare_models(X_train, y_train):
    """Return five-fold cross-validation metrics for every model."""
    results = []

    for name, model in get_candidate_models().items():
        pipeline = build_pipeline(model)

        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=CV,
            scoring=SCORING,
            n_jobs=-1,
            error_score="raise",
        )

        row = {"model": name}

        for metric in SCORING:
            values = scores[f"test_{metric}"]
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std())

        results.append(row)

    return pd.DataFrame(results)


# Search eight hyperparameter combinations.
def tune_model(X_train, y_train):
    """Tune Logistic Regression using ROC-AUC scoring."""
    model = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE,
    )

    pipeline = build_pipeline(model)

    parameter_grid = {
        "model__C": [0.01, 0.1, 1.0, 10.0],
        "model__fit_intercept": [True, False],
    }

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=parameter_grid,
        scoring="roc_auc",
        cv=CV,
        n_jobs=-1,
        refit=True,
        error_score="raise",
    )

    search.fit(X_train, y_train)
    return search


# Find the cheapest threshold using out-of-fold training probabilities.
def select_call_threshold(X_train, y_train, tuned_pipeline):
    """Select a threshold without using the held-out test set."""
    probabilities = cross_val_predict(
        tuned_pipeline,
        X_train,
        y_train,
        cv=CV,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]

    threshold, training_cost = choose_threshold(
        y_train,
        probabilities,
    )

    return threshold, training_cost


# Demonstrate how an after-the-visit column causes data leakage.
def run_leakage_experiment(X_train, y_train, fee_charged, best_params):
    """Compare honest and leaking feature sets using training CV only."""

    # Build the same selected classifier for both comparisons.
    model = LogisticRegression(
        C=best_params["model__C"],
        fit_intercept=best_params["model__fit_intercept"],
        max_iter=1000,
        random_state=RANDOM_STATE,
    )

    # Honest model: approved booking-time features only.
    honest_pipeline = build_pipeline(model)

    honest_scores = cross_validate(
        honest_pipeline,
        X_train,
        y_train,
        cv=CV,
        scoring="roc_auc",
        n_jobs=-1,
        error_score="raise",
    )

    # Leaking experiment: add the after-the-visit fee column.
    leaking_features = NUMERIC_FEATURES + ["fee_charged"]

    leaking_pipeline = build_pipeline(
        model,
        numeric_features=leaking_features,
        categorical_features=CATEGORICAL_FEATURES,
    )

    X_leaking = X_train.copy()
    X_leaking["fee_charged"] = fee_charged.reindex(X_train.index)

    leaking_scores = cross_validate(
        leaking_pipeline,
        X_leaking,
        y_train,
        cv=CV,
        scoring="roc_auc",
        n_jobs=-1,
        error_score="raise",
    )

    return {
        "honest_roc_auc": float(honest_scores["test_score"].mean()),
        "leaking_roc_auc": float(leaking_scores["test_score"].mean()),
    }


# Calculate classification metrics on the held-out test set.
def evaluate_test_set(pipeline, X_test, y_test, threshold):
    """Evaluate the final fitted model once on unseen test data."""
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    decisions = (probabilities >= threshold).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_test, decisions)),
        "precision": float(precision_score(y_test, decisions, zero_division=0)),
        "recall": float(recall_score(y_test, decisions)),
        "f1": float(f1_score(y_test, decisions)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
    }

    costs = compare_costs(
        y_test,
        probabilities,
        threshold,
    )

    metrics["costs"] = costs

    # Positive savings mean the model is cheaper than the baseline.
    metrics["savings_vs_call_nobody"] = costs["call_nobody"] - costs["model"]
    metrics["savings_vs_call_everybody"] = costs["call_everybody"] - costs["model"]

    metrics["beats_both_baselines"] = bool(
        costs["model"] < costs["call_nobody"]
        and costs["model"] < costs["call_everybody"]
    )

    return metrics


# Estimate uncertainty in cost differences on the fixed test set.
def bootstrap_cost_savings(y_test, probabilities, threshold):
    """Return approximate 95% bootstrap intervals for cost savings."""
    actual = np.asarray(y_test)
    scores = np.asarray(probabilities)
    rng = np.random.default_rng(RANDOM_STATE)

    savings_nobody = []
    savings_everybody = []

    for _ in range(1000):
        indices = rng.integers(0, len(actual), size=len(actual))

        costs = compare_costs(
            actual[indices],
            scores[indices],
            threshold,
        )

        savings_nobody.append(costs["call_nobody"] - costs["model"])
        savings_everybody.append(costs["call_everybody"] - costs["model"])

    return {
        "vs_call_nobody": [
            float(value) for value in np.percentile(savings_nobody, [2.5, 97.5])
        ],
        "vs_call_everybody": [
            float(value) for value in np.percentile(savings_everybody, [2.5, 97.5])
        ],
    }


# Save a Python dictionary as readable JSON.
def save_json(path, content):
    """Write structured results to a JSON file."""
    with path.open("w", encoding="utf-8") as file:
        json.dump(content, file, indent=4)
        file.write("\n")


# Run the complete reproducible training process.
def main():
    """Train, evaluate once, and recreate all required output files."""

    # Create output folders if they do not exist.
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Generate the instructor's original dataset.
    frame = make_appointments()

    # Reserve the held-out test set until all model decisions are fixed.
    X_train, X_test, y_train, y_test = split_data(frame)

    print("\nTraining records:", len(X_train))
    print("Testing records:", len(X_test))

    # Step 1: Compare all candidate models.
    comparison = compare_models(X_train, y_train)

    print("\nFive-fold cross-validation results:")
    print(comparison.round(4).to_string(index=False))

    # Save required cross-validation comparison report.
    comparison.to_csv(REPORT_DIR / "cv_results.csv", index=False)

    # Step 2: Tune using training data only.
    search = tune_model(X_train, y_train)

    original = comparison.loc[comparison["model"] == "logistic_regression"].iloc[0]

    original_auc = float(original["roc_auc_mean"])
    original_std = float(original["roc_auc_std"])
    improvement = float(search.best_score_ - original_auc)

    tuning_info = {
        "scoring": "roc_auc",
        "combinations": len(search.cv_results_["params"]),
        "best_params": search.best_params_,
        "untuned_roc_auc": original_auc,
        "untuned_std": original_std,
        "tuned_roc_auc": float(search.best_score_),
        "improvement": improvement,
        "improvement_exceeds_spread": bool(improvement > original_std),
    }

    print("\nGridSearchCV results:")
    print("Best parameters:", search.best_params_)
    print("Best CV ROC-AUC:", round(search.best_score_, 4))

    # Step 3: Select threshold from out-of-fold training predictions.
    threshold, training_cost = select_call_threshold(
        X_train,
        y_train,
        search.best_estimator_,
    )

    print("\nSelected threshold:", threshold)
    print("Out-of-fold training cost: $", training_cost, sep="")

    # Step 4: Demonstrate data leakage using training data only.
    leakage = run_leakage_experiment(
        X_train,
        y_train,
        frame.loc[X_train.index, "fee_charged"],
        search.best_params_,
    )

    print("\nLeakage comparison:")
    print("Honest ROC-AUC:", round(leakage["honest_roc_auc"], 4))
    print("Leaking ROC-AUC:", round(leakage["leaking_roc_auc"], 4))

    # Step 5: Use the final pipeline fitted on all training records.
    final_pipeline = search.best_estimator_

    # Step 6: Evaluate the held-out test data after all choices are fixed.
    test_results = evaluate_test_set(
        final_pipeline,
        X_test,
        y_test,
        threshold,
    )

    # Estimate uncertainty using the same fixed test predictions.
    test_probabilities = final_pipeline.predict_proba(X_test)[:, 1]

    test_results["savings_95pct_bootstrap_interval"] = bootstrap_cost_savings(
        y_test,
        test_probabilities,
        threshold,
    )

    print("\nFinal held-out test results:")
    for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        print(f"{metric}: {test_results[metric]:.4f}")

    print("\nTest-set business costs:")
    for strategy, cost in test_results["costs"].items():
        print(f"{strategy}: ${cost}")

    print(
        "Model beats both baselines:",
        test_results["beats_both_baselines"],
    )

    # Step 7: Save the complete fitted preprocessing/model pipeline.
    joblib.dump(final_pipeline, MODEL_DIR / "no_show.joblib")

    # Store settings needed by the prediction class and deployment.
    model_info = {
        "model": "LogisticRegression",
        "scikit_learn_version": sklearn.__version__,
        "seed": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "threshold": threshold,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "features": FEATURES,
        "best_settings": search.best_params_,
        "test_results": test_results,
    }

    save_json(MODEL_DIR / "model_info.json", model_info)

    # Save the report including model selection and business costs.
    report = {
        "model_comparison": comparison.to_dict(orient="records"),
        "tuning": tuning_info,
        "out_of_fold_training_cost": training_cost,
        "selected_threshold": threshold,
        "leakage_comparison": leakage,
        "test_results": test_results,
    }

    save_json(REPORT_DIR / "metrics.json", report)

    print("\nModel and reports saved successfully.")


if __name__ == "__main__":
    main()
