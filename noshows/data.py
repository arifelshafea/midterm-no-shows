"""Invented physiotherapy appointments, for the Midterm Project."""

import numpy as np
import pandas as pd


def make_appointments(n: int = 4000, seed: int = 11) -> pd.DataFrame:
    """Return n invented appointments. no_show is 1 for a missed visit."""
    rng = np.random.default_rng(seed)
    lead_days = rng.gamma(2.0, 6.0, n).clip(0, 60).round().astype(int)
    age = np.round(rng.normal(46, 15, n).clip(16, 92)).astype(int)
    prior_no_shows = rng.poisson(0.5, n).clip(0, 6)
    hour = rng.choice(np.arange(8, 18), n)
    weekday = rng.choice(["mon", "tue", "wed", "thu", "fri"], n)
    booking_kinds = ["online", "phone", "front_desk"]
    booking = rng.choice(booking_kinds, n, p=[0.45, 0.35, 0.20])
    plans = ["private", "provincial", "self_pay"]
    insurance = rng.choice(plans, n, p=[0.5, 0.3, 0.2])
    kinds = ["physio", "massage", "acupuncture"]
    treatment = rng.choice(kinds, n, p=[0.6, 0.25, 0.15])
    first_visit = (rng.random(n) < 0.22).astype(int)
    distance = np.round(rng.gamma(2.0, 4.0, n).clip(0.5, 60), 1)
    early_or_late = (hour <= 9) | (hour >= 16)
    score = (
        -3.3
        + 0.055 * lead_days
        + 0.75 * prior_no_shows
        + 0.45 * early_or_late
        + 0.40 * (weekday == "fri")
        + 0.60 * (weekday == "mon")
        + 0.80 * (booking == "online")
        - 0.30 * (booking == "front_desk")
        + 0.60 * (insurance == "self_pay")
        - 0.25 * (insurance == "provincial")
        + 0.55 * first_visit
        + 0.035 * distance
        - 0.016 * (age - 46)
        - 0.30 * (treatment == "acupuncture")
    )
    no_show = (rng.random(n) < 1 / (1 + np.exp(-score))).astype(int)
    fee = np.where(no_show == 1, (rng.random(n) < 0.7).astype(int), 0)
    frame = pd.DataFrame(
        {
            "appointment_id": np.arange(50000, 50000 + n),
            "lead_days": lead_days,
            "age": age,
            "prior_no_shows": prior_no_shows,
            "hour": hour,
            "weekday": weekday,
            "booking": booking,
            "insurance": insurance,
            "treatment": treatment,
            "first_visit": first_visit,
            "distance_km": distance,
            "fee_charged": fee,
            "no_show": no_show,
        }
    )
    # some values are not recorded
    frame.loc[rng.random(n) < 0.05, "distance_km"] = np.nan
    return frame