import math

import numpy as np
import pandas as pd
import pytest

from research.pregame_risk_shadow import (
    evidence_support,
    normalized_entropy,
    normalized_js_divergence,
    normalized_margin,
    score_case,
    score_frame,
)


def test_entropy_is_low_for_certain_and_high_for_uniform():
    assert normalized_entropy([1.0, 0.0]) < 1e-6
    assert math.isclose(normalized_entropy([0.5, 0.5]), 1.0, rel_tol=1e-6)
    assert math.isclose(normalized_entropy([1 / 3] * 3), 1.0, rel_tol=1e-6)


def test_margin_risk_tracks_ambiguity():
    assert normalized_margin([0.99, 0.01]) < normalized_margin([0.60, 0.40])


def test_js_divergence_is_zero_for_identical_experts():
    assert normalized_js_divergence([[0.7, 0.3], [0.7, 0.3]]) < 1e-9


def test_js_divergence_is_higher_for_conflicting_experts():
    same = normalized_js_divergence([[0.8, 0.2], [0.75, 0.25]])
    conflict = normalized_js_divergence([[0.99, 0.01], [0.01, 0.99]])
    assert conflict > same


def test_case_score_uses_expert_disagreement_and_is_bounded():
    row = {
        "pred_home": 0.55,
        "pred_away": 0.45,
        "expert_one_home": 0.95,
        "expert_one_away": 0.05,
        "expert_two_home": 0.05,
        "expert_two_away": 0.95,
        "confirmed_starters": True,
        "weather_pit_safe": True,
        "ctx_lineup_present": True,
        "training_included": True,
    }
    out = score_case(row)
    assert out["risk_available"] is True
    assert out["expert_count"] == 2
    assert out["expert_disagreement"] > 0.7
    assert 0.0 <= out["risk_score"] <= 1.0


def test_missing_forecast_is_fail_closed():
    out = score_case({"pred_home": np.nan, "pred_away": np.nan})
    assert out["risk_available"] is False
    assert math.isnan(out["risk_score"])


def test_evidence_support_is_explicit():
    support, count = evidence_support(
        {
            "confirmed_starters": True,
            "starter_confirmation_state": "CONFIRMED",
            "weather_pit_safe": False,
            "ctx_lineup_present": True,
        }
    )
    assert count == 4
    assert math.isclose(support, 0.75)


def test_score_frame_does_not_change_forecast_columns():
    df = pd.DataFrame(
        [
            {
                "game_id": "1",
                "pred_home": 0.7,
                "pred_away": 0.3,
                "training_included": True,
            }
        ]
    )
    out = score_frame(df)
    assert out.loc[0, "pred_home"] == df.loc[0, "pred_home"]
    assert out.loc[0, "pred_away"] == df.loc[0, "pred_away"]
    assert bool(out.loc[0, "risk_research_only"]) is True


def test_three_class_distribution_is_supported():
    row = {
        "pred_home": 0.5,
        "pred_draw": 0.2,
        "pred_away": 0.3,
        "expert_a_home": 0.6,
        "expert_a_draw": 0.1,
        "expert_a_away": 0.3,
        "expert_b_home": 0.4,
        "expert_b_draw": 0.3,
        "expert_b_away": 0.3,
    }
    out = score_case(row)
    assert out["risk_available"] is True
    assert out["expert_count"] == 2
