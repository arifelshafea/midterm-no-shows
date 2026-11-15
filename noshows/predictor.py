"""Turn a booking into a chance of a no-show and an action."""

import json
from dataclasses import dataclass
from pathlib import Path

import joblib

from noshows.appointment import Appointment

# The models folder sits next to the noshows package.
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"


@dataclass(frozen=True)
class Prediction:
    """The chance of a no-show, the threshold used, and the action."""

    probability: float
    threshold: float
    call_patient: bool

    @property
    def action(self):
        """A plain-language action for the front desk."""
        if self.call_patient:
            return "Call the patient the day before to confirm."
        return "No call needed."


class NoShowPredictor:
    """Load the saved model once, then score bookings."""

    def __init__(self, model_dir=MODEL_DIR):
        model_dir = Path(model_dir)

        # Load the fitted pipeline and its information once, here.
        self.pipeline = joblib.load(model_dir / "no_show.joblib")
        with open(model_dir / "model_info.json", encoding="utf-8") as file:
            self.info = json.load(file)

        # The threshold belongs to the model, so it is read, never typed.
        self.threshold = self.info["threshold"]

    def should_call(self, probability):
        """Return True when the probability reaches the threshold."""
        return probability >= self.threshold

    def predict(self, appointment):
        """Return the probability and the decision together."""
        if not isinstance(appointment, Appointment):
            raise TypeError("predict needs an Appointment.")

        row = appointment.to_frame()
        probability = float(self.pipeline.predict_proba(row)[0][1])

        return Prediction(
            probability=probability,
            threshold=self.threshold,
            call_patient=self.should_call(probability),
        )

    def __repr__(self):
        return (
            f"NoShowPredictor(model={self.info['model']}, "
            f"threshold={self.threshold}, features={len(self.info['features'])})"
        )
