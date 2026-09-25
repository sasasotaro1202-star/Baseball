import math

import pytest

from research.case_risk_layer import assess_case_risk, normalized_entropy, top_gap_risk


def test_entropy_and_gap_are_bounded_and_ordered():
    assert normalized_entropy([0.5, 0.5]) == pytest.approx(1.0)
    assert top_gap_risk([0.99, 0.01]) < top_gap_risk([0.60, 0.40])


def test_case_risk_never_reverses_prediction():
    out = assess_case_risk(
        [0.72, 0.18, 0.10],
        disagreement=0.8,
        conformal=0.8,
        drift=0.8,
        lineup_state="UNKNOWN",
        starter_state="VERIFIED",
        weather_state="PROJECTED",
    )
    assert out["research_only"] is True
    assert 0.0 <= out["risk_score"] <= 1.0
    assert 0.0 <= out["upset_risk_signal"] <= 1.0
    assert out["risk_state"] in {"LOW", "MEDIUM", "HIGH"}



def test_case_risk_flags_situational_anomaly():
    out = assess_case_risk(
        [0.72, 0.28],
        rest_travel={
            "home": {"rest_days": 1.0, "games_last_3d": 3, "games_last_7d": 6, "travel_miles": 1200},
            "away": {"rest_days": 5.0, "games_last_3d": 1, "games_last_7d": 3, "travel_miles": 50},
        },
    )
    assert out["situational_risk"] >= 0.60
    assert "situational_load_or_travel_anomaly" in out["reasons"]
