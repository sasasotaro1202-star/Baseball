import pandas as pd
import pytest

from monitoring.result_audit import audit_predictions


def _npb_predictions():
    return pd.DataFrame([{
        "event_id": "g1",
        "home_win_probability": 0.55,
        "draw_probability": 0.20,
        "away_win_probability": 0.25,
    }])


def _npb_results():
    return pd.DataFrame([{
        "event_id": "g1",
        "actual_home_score": 3,
        "actual_away_score": 2,
    }])


def test_result_audit_rejects_duplicate_prediction_event_ids():
    p = pd.concat([_npb_predictions(), _npb_predictions()], ignore_index=True)
    with pytest.raises(ValueError):
        audit_predictions(p, _npb_results(), league="NPB")


def test_result_audit_rejects_duplicate_result_event_ids():
    r = pd.concat([_npb_results(), _npb_results()], ignore_index=True)
    with pytest.raises(ValueError):
        audit_predictions(_npb_predictions(), r, league="NPB")


def test_result_audit_rejects_probability_rows_that_do_not_sum_to_one():
    p = _npb_predictions()
    p.loc[0, "away_win_probability"] = 0.35
    with pytest.raises(ValueError):
        audit_predictions(p, _npb_results(), league="NPB")


def test_result_audit_reports_unmatched_predictions_without_scoring_them():
    p = pd.concat([_npb_predictions(), pd.DataFrame([{
        "event_id": "missing",
        "home_win_probability": 0.5,
        "draw_probability": 0.2,
        "away_win_probability": 0.3,
    }])], ignore_index=True)
    report = audit_predictions(p, _npb_results(), league="NPB")
    assert report["matched_rows"] == 1
    assert report["unmatched_predictions"] == 1
