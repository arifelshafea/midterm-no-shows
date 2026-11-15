"""Tests for the Gradio prediction interface."""

from app import predict_no_show


def test_prediction_returns_three_outputs():
    """A valid appointment should return three outputs."""
    result = predict_no_show(
        lead_days=7,
        age=35,
        prior_no_shows=0,
        hour=9,
        weekday="mon",
        booking="online",
        insurance="private",
        treatment="physio",
        first_visit=False,
        distance_km=10.0,
    )

    assert len(result) == 3


def test_invalid_age_shows_error():
    """An invalid age should return a clear error message."""
    probability, action, warning = predict_no_show(
        lead_days=7,
        age=5,
        prior_no_shows=0,
        hour=9,
        weekday="mon",
        booking="online",
        insurance="private",
        treatment="physio",
        first_visit=False,
        distance_km=10.0,
    )

    assert probability == ""
    assert action == ""
    assert "age" in warning.lower()
    assert "12 and 110" in warning


def test_missing_distance_shows_warning():
    """A missing distance should still allow a prediction."""
    probability, action, warning = predict_no_show(
        lead_days=7,
        age=35,
        prior_no_shows=0,
        hour=9,
        weekday="mon",
        booking="online",
        insurance="private",
        treatment="physio",
        first_visit=False,
        distance_km=None,
    )

    assert probability.endswith("%")
    assert action in ["Call Patient", "No Call Needed"]
    assert "Distance was not provided" in warning


def test_invalid_first_visit_shows_error():
    """A fractional first_visit value must be rejected."""
    probability, action, warning = predict_no_show(
        lead_days=7,
        age=35,
        prior_no_shows=0,
        hour=9,
        weekday="mon",
        booking="online",
        insurance="private",
        treatment="physio",
        first_visit=0.5,
        distance_km=10.0,
    )

    assert probability == ""
    assert action == ""
    assert "first_visit" in warning
