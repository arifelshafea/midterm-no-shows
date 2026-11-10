"""Feature selection, data splitting, and model pipeline construction."""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Keep the random seed and split size in one place.
RANDOM_STATE = 42
TEST_SIZE = 0.25

# Numeric features available when an appointment is booked.
NUMERIC_FEATURES = [
    "lead_days",
    "age",
    "prior_no_shows",
    "hour",
    "first_visit",
    "distance_km",
]

# Categorical features available when an appointment is booked.
CATEGORICAL_FEATURES = [
    "weekday",
    "booking",
    "insurance",
    "treatment",
]

# The target column is never used as a prediction feature.
TARGET = "no_show"

# Use the same approved features throughout the project.
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# Split appointments into stratified training and testing sets.
def split_data(frame):
    """Create reproducible 75/25 stratified training and test sets."""
    X = frame[FEATURES]
    y = frame[TARGET]

    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )


# Build one reusable preprocessing and prediction pipeline.
def build_pipeline(model, numeric_features=None, categorical_features=None):
    """Build a pipeline with configurable training feature lists."""

    # Use the approved features unless an experiment supplies alternatives.
    if numeric_features is None:
        numeric_features = NUMERIC_FEATURES

    if categorical_features is None:
        categorical_features = CATEGORICAL_FEATURES

    # Replace missing numeric values using the training-set median.
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Encode categories and tolerate unseen values during transformation.
    categorical_pipeline = Pipeline(
        steps=[
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # Apply the correct preprocessing to each feature group.
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )

    # Combine preprocessing and classification into one fitted pipeline.
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )
