import dataclasses
import math

import pytest

from noshows.appointment import Appointment
from noshows.pipeline import FEATURES

VALID = {
    "lead_days": 5,
    "age": 40,
    "prior_no_shows": 0,
    "hour": 10,
    "first_visit": 0,
    "weekday": "mon",
    "booking": "online",
    "insurance": "private",
    "treatment": "physio",
    "distance_km": 7.5,
}


def make(**changes):
    return Appointment(**{**VALID, **changes})


def test_valid_booking_is_accepted():
    appointment = make()
    assert appointment.age == 40
    assert appointment.weekday == "mon"
    assert appointment.distance_missing is False


def test_empty_distance_is_accepted_and_flagged():
    appointment = make(distance_km=None)
    assert appointment.distance_missing is True


@pytest.mark.parametrize(
    "field, value",
    [
        ("lead_days", 0),
        ("lead_days", 120),
        ("age", 12),
        ("age", 110),
        ("prior_no_shows", 0),
        ("prior_no_shows", 30),
        ("hour", 8),
        ("hour", 17),
        ("distance_km", 0.1),
        ("distance_km", 150),
        ("first_visit", 0),
        ("first_visit", 1),
    ],
)
def test_limits_themselves_are_accepted(field, value):
    make(**{field: value})


@pytest.mark.parametrize(
    "field, value",
    [
        ("lead_days", -1),
        ("lead_days", 121),
        ("age", 11),
        ("age", 111),
        ("age", "40"),
        ("prior_no_shows", -1),
        ("prior_no_shows", 31),
        ("hour", 7),
        ("hour", 18),
        ("distance_km", 0.05),
        ("distance_km", 151),
        ("distance_km", float("nan")),
        ("distance_km", float("inf")),
        ("first_visit", 2),
        ("first_visit", True),
    ],
)
def test_impossible_numbers_are_rejected_naming_the_field(field, value):
    with pytest.raises(ValueError, match=field):
        make(**{field: value})


@pytest.mark.parametrize(
    "field, value",
    [
        ("weekday", "Mon"),
        ("weekday", "sat"),
        ("booking", "Online"),
        ("booking", "fax"),
        ("insurance", "Private"),
        ("insurance", "none"),
        ("treatment", "Physio"),
        ("treatment", "yoga"),
    ],
)
def test_unseen_categories_are_rejected_naming_the_field(field, value):
    with pytest.raises(ValueError, match=field):
        make(**{field: value})


@pytest.mark.parametrize("empty", [None, ""])
@pytest.mark.parametrize("field", [name for name in FEATURES if name != "distance_km"])
def test_required_fields_cannot_be_empty(field, empty):
    with pytest.raises(ValueError, match=field):
        make(**{field: empty})


def test_appointment_cannot_be_changed_after_creation():
    appointment = make()
    with pytest.raises(dataclasses.FrozenInstanceError):
        appointment.age = 99


def test_to_frame_has_one_row_with_training_columns():
    frame = make().to_frame()
    assert list(frame.columns) == FEATURES
    assert len(frame) == 1
    assert frame.loc[0, "age"] == 40


def test_to_frame_fills_missing_distance_with_nan():
    frame = make(distance_km=None).to_frame()
    assert math.isnan(frame.loc[0, "distance_km"])
