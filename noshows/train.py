
"""Compare, tune, and select a threshold for no-show prediction."""

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
)

from noshows.data import make_appointments
from noshows.evaluation import choose_threshold
from noshows.pipeline import RANDOM_STATE, build_pipeline, split_data

# Required metrics for comparing classification models.
SCORING = {
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
    "accuracy": "accuracy",
}

# Reproducible five-fold stratified cross-validation.
CV = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE,
)


# Define the baseline and three candidate classifiers.
def get_candidate_models():
    """Return models for the initial cross-validation comparison."""
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


# Compare all models using training data only.
def compare_models(X_train, y_train):
    """Return mean and standard deviation of cross-validation metrics."""
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


# Tune Logistic Regression using training data only.
def tune_model(X_train, y_train):
    """Search eight Logistic Regression hyperparameter combinations."""
    model = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE,
    )

    pipeline = build_pipeline(model)

    # Four regularization values and two intercept settings.
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


# Generate out-of-fold probabilities and optimize the calling threshold.
def select_call_threshold(X_train, y_train, tuned_pipeline):
    """Choose the lowest-cost threshold using out-of-fold probabilities."""

    # cross_val_predict fits fresh pipeline clones in each fold.
    # Every row is predicted by a model that did not train on that row.
    probabilities = cross_val_predict(
        tuned_pipeline,
        X_train,
        y_train,
        cv=CV,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]

    # Use the cost function defined in evaluation.py.
    threshold, training_cost = choose_threshold(
        y_train,
        probabilities,
    )

    return threshold, training_cost


# Run the comparison, tuning, and threshold optimization.
def main():
    """Select a model and threshold without evaluating the test set."""
    frame = make_appointments()

    # Reserve the test data for the final evaluation.
    X_train, _, y_train, _ = split_data(frame)

    # Step 1: Compare candidate models.
    results = compare_models(X_train, y_train)

    print("\nFive-fold cross-validation results:")
    print(results.round(4).to_string(index=False))

    # Step 2: Tune Logistic Regression.
    search = tune_model(X_train, y_train)

    print("\nGridSearchCV results:")
    print("Scoring metric: ROC-AUC")
    print("Hyperparameter combinations:", len(search.cv_results_["params"]))
    print("Best parameters:", search.best_params_)
    print("Best cross-validation ROC-AUC:", round(search.best_score_, 4))

    # Step 3: Compare tuned and untuned performance.
    original = results.loc[
        results["model"] == "logistic_regression"
    ].iloc[0]

    original_auc = float(original["roc_auc_mean"])
    original_std = float(original["roc_auc_std"])
    improvement = float(search.best_score_ - original_auc)

    print("\nTuning comparison:")
    print("Untuned ROC-AUC:", round(original_auc, 4))
    print("Untuned ROC-AUC standard deviation:", round(original_std, 4))
    print("Tuned ROC-AUC:", round(search.best_score_, 4))
    print("ROC-AUC improvement:", round(improvement, 4))
    print(
        "Improvement exceeds untuned fold-to-fold spread:",
        improvement > original_std,
    )

    # Step 4: Choose the calling threshold from out-of-fold predictions.
    threshold, training_cost = select_call_threshold(
        X_train,
        y_train,
        search.best_estimator_,
    )

    print("\nOut-of-fold threshold optimization:")
    print("Selected threshold:", threshold)
    print("Training out-of-fold model cost: $", training_cost, sep="")

    # The test set is not evaluated here.


if __name__ == "__main__":
    main()
