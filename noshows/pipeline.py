
"""Feature selection, data splitting, and model pipeline construction."""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Keep the random seed and split size in one place.
RANDOM_STATE = 42
TEST_SIZE = 0.25

# These numeric values are known when an appointment is booked.
NUMERIC_FEATURES = [
    "lead_days",
    "age",
    "prior_no_shows",
    "hour",
    "first_visit",
    "distance_km",
]

# These categorical values are also known when the appointment is booked.
CATEGORICAL_FEATURES = [
    "weekday",
    "booking",
    "insurance",
    "treatment",
]

# The target is used for training, not prediction inputs.
TARGET = "no_show"

# Keep feature lists consistent across training, classes, and Gradio.
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# Split the data while preserving the no-show proportion in both sets.
def split_data(frame):
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
def build_pipeline(model):
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )
