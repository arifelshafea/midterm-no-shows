import json
from pathlib import Path

import joblib
import pytest

from noshows.appointment import Appointment
from noshows.predictor import NoShowPredictor

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"

SAFE = {
    "lead_days": 1,
    "age": 60,
    "prior_no_shows": 0,
    "hour": 12,
    "first_visit": 0,
    "weekday": "wed",
    "booking": "front_desk",
    "insurance": "provincial",
    "treatment": "acupuncture",
    "distance_km": 2,
}

RISKY = {
    "lead_days": 45,
    "age": 22,
    "prior_no_shows": 4,
    "hour": 9,
    "first_visit": 1,
    "weekday": "fri",
    "booking": "online",
    "insurance": "self_pay",
    "treatment": "physio",
    "distance_km": 30,
}


@pytest.fixture(scope="module")
def predictor():
    return NoShowPredictor()


def test_probability_is_between_0_and_1(predictor):
    probability = predictor.predict(Appointment(**SAFE)).probability
    assert 0.0 <= probability <= 1.0


def test_riskier_booking_scores_higher(predictor):
    safe = predictor.predict(Appointment(**SAFE)).probability
    risky = predictor.predict(Appointment(**RISKY)).probability
    assert risky > safe


def test_missing_distance_still_gets_a_prediction(predictor):
    booking = Appointment(**{**SAFE, "distance_km": None})
    assert 0.0 <= predictor.predict(booking).probability <= 1.0


@pytest.mark.parametrize(
    "difference, expected",
    [
        (-0.01, False),  # just below the threshold: no call
        (0.0, True),  # exactly at the threshold: call
        (0.01, True),  # just above the threshold: call
    ],
)
def test_decision_around_the_threshold(predictor, difference, expected):
    probability = predictor.threshold + difference
    assert predictor.should_call(probability) is expected


def test_threshold_comes_from_model_info(predictor):
    info = json.loads((MODEL_DIR / "model_info.json").read_text(encoding="utf-8"))
    assert predictor.threshold == info["threshold"]


def test_prediction_carries_probability_threshold_and_decision(predictor):
    prediction = predictor.predict(Appointment(**RISKY))
    assert prediction.threshold == predictor.threshold
    assert prediction.call_patient == (prediction.probability >= predictor.threshold)


def test_action_matches_the_decision(predictor):
    risky = predictor.predict(Appointment(**RISKY))
    safe = predictor.predict(Appointment(**SAFE))
    assert risky.action != safe.action


@pytest.mark.parametrize("bad", [{"age": 40}, None, "booking", 42])
def test_predict_only_accepts_an_appointment(predictor, bad):
    with pytest.raises(TypeError):
        predictor.predict(bad)


def test_repr_is_readable(predictor):
    text = repr(predictor)
    assert text.startswith("NoShowPredictor(")
    assert str(predictor.threshold) in text


def test_model_file_is_loaded_once(monkeypatch):
    original_load = joblib.load
    calls = 0

    def counting_load(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_load(*args, **kwargs)

    monkeypatch.setattr(joblib, "load", counting_load)

    predictor = NoShowPredictor()
    predictor.predict(Appointment(**SAFE))
    predictor.predict(Appointment(**RISKY))

    assert calls == 1
