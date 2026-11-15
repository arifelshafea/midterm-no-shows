"""One booking, as the front desk enters it."""

import math
from dataclasses import dataclass
from numbers import Real

import pandas as pd

from noshows.pipeline import FEATURES

# Allowed range for each number: (lowest, highest).
NUMBER_LIMITS = {
    "lead_days": (0, 120),
    "age": (12, 110),
    "prior_no_shows": (0, 30),
    "hour": (8, 17),
    "distance_km": (0.1, 150),
}

# Allowed values for each category.
CHOICES = {
    "weekday": ("mon", "tue", "wed", "thu", "fri"),
    "booking": ("online", "phone", "front_desk"),
    "insurance": ("private", "provincial", "self_pay"),
    "treatment": ("physio", "massage", "acupuncture"),
}


def _is_number(value):
    """Return True for a real, finite number (True/False do not count)."""
    return (
        isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)
    )


@dataclass(frozen=True, kw_only=True)
class Appointment:
    """One booking. Values are checked on creation and cannot change."""

    lead_days: float
    age: float
    prior_no_shows: float
    hour: float
    first_visit: int
    weekday: str
    booking: str
    insurance: str
    treatment: str
    distance_km: float | None = None  # empty is the one accepted gap

    def __post_init__(self):
        """Raise ValueError, naming the field, for anything impossible."""

        # Every field is required except distance_km.
        for field in FEATURES:
            value = getattr(self, field)
            if field != "distance_km" and (value is None or value == ""):
                raise ValueError(f"{field}: This field is required.")

        # Numbers must be real and inside their limits.
        for field, (low, high) in NUMBER_LIMITS.items():
            value = getattr(self, field)
            if value is None and field == "distance_km":
                continue
            if not _is_number(value) or not low <= value <= high:
                raise ValueError(f"{field}: Must be a number between {low} and {high}.")

        # first_visit is a flag: 0 or 1 only.
        if isinstance(self.first_visit, bool) or self.first_visit not in (0, 1):
            raise ValueError("first_visit: Must be 0 or 1.")

        # Categories must be ones the model was trained on.
        for field, allowed in CHOICES.items():
            if getattr(self, field) not in allowed:
                raise ValueError(f"{field}: Must be one of {', '.join(allowed)}.")

    @property
    def distance_missing(self):
        """True when distance_km was left empty (the page shows a warning)."""
        return self.distance_km is None

    def to_frame(self):
        """Return a one-row table using the training column names."""
        row = {name: getattr(self, name) for name in FEATURES}
        if row["distance_km"] is None:
            row["distance_km"] = math.nan  # the pipeline fills this in
        return pd.DataFrame([row], columns=FEATURES)
