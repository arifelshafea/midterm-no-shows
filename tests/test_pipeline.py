"""Tests for data generation, feature selection, and preprocessing."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from noshows.data import make_appointments
from noshows.pipeline import (
    FEATURES,
    RANDOM_STATE,
    build_pipeline,
    split_data,
)


# Verify the instructor's dataset is reproducible with the same seed.
def test_data_is_reproducible():
    first = make_appointments()
    second = make_appointments()

    pd.testing.assert_frame_equal(first, second)


# Verify that identifiers, leakage columns, and target are excluded.
@pytest.mark.parametrize(
    "excluded_column",
    ["appointment_id", "fee_charged", "no_show"],
)
def test_excluded_columns_are_not_features(excluded_column):
    assert excluded_column not in FEATURES


# Verify the required 75/25 split.
def test_split_sizes():
    frame = make_appointments()
    X_train, X_test, y_train, y_test = split_data(frame)

    assert len(X_train) == 3000
    assert len(X_test) == 1000
    assert len(y_train) == 3000
    assert len(y_test) == 1000


# Verify that stratification preserves the no-show proportion.
def test_split_is_stratified():
    frame = make_appointments()
    _, _, y_train, y_test = split_data(frame)

    original_rate = frame["no_show"].mean()

    assert abs(y_train.mean() - original_rate) < 0.01
    assert abs(y_test.mean() - original_rate) < 0.01


# Verify that splitting is reproducible.
def test_split_is_reproducible():
    frame = make_appointments()

    first = split_data(frame)
    second = split_data(frame)

    for first_part, second_part in zip(first, second, strict=True):
        pd.testing.assert_index_equal(first_part.index, second_part.index)


# Verify the preprocessing pipeline handles missing distance values.
def test_pipeline_handles_missing_distance():
    frame = make_appointments(n=200, seed=RANDOM_STATE)
    frame.loc[0, "distance_km"] = np.nan

    X_train, X_test, y_train, _ = split_data(frame)

    pipeline = build_pipeline(
        LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    )
    pipeline.fit(X_train, y_train)

    probabilities = pipeline.predict_proba(X_test)[:, 1]

    assert len(probabilities) == len(X_test)
    assert np.all(np.isfinite(probabilities))
    assert np.all((probabilities >= 0) & (probabilities <= 1))
