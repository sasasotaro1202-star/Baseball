import pandas as pd

from core.pit import assert_no_future_rows, filter_as_of
from evaluation.metrics import classification_metrics, score_metrics
from research.adoption_gate import candidate_lock, evaluate_locked_holdout


def test_pit_excludes_future_available_rows():
    df = pd.DataFrame({
        "event_id": ["g1", "g2"],
        "event_time": ["2026-09-01T00:00:00Z", "2026-09-02T00:00:00Z"],
        "available_ts": ["2026-08-31T00:00:00Z", "2026-09-02T12:00:00Z"],
    })
    result = filter_as_of(df, "2026-09-02T06:00:00Z")
    assert result.rows_in == 2
    assert result.rows_out == 1
    assert result.frame.iloc[0]["event_id"] == "g1"


def test_metrics_and_gate():
    metrics = classification_metrics([0, 1, 1], [[0.8, 0.2], [0.3, 0.7], [0.4, 0.6]], classes=[0, 1])
    assert metrics["rows"] == 3
    score = score_metrics([3, 2], [2, 4], [2.5, 2], [2, 3.5])
    assert score["ScoreMAE"] >= 0

    locked = candidate_lock(development_metrics={"LogLoss": 0.60, "rows": 300}, candidate_id="cand-001")
    assert locked["stage"] == "candidate_locked"
    result = evaluate_locked_holdout(
        {"LogLoss": 0.60, "Brier": 0.20, "Accuracy": 0.70},
        {"LogLoss": 0.58, "Brier": 0.19, "Accuracy": 0.705, "rows": 300},
        validation_windows=2,
        calibration_ok=True,
        no_future_target_data=True,
        reproducible=True,
    )
    assert result["adopt"] is True


def test_pit_assertion_detects_future_availability():
    df = pd.DataFrame({"available_ts": ["2026-09-03T00:00:00Z"]})
    try:
        assert_no_future_rows(df, "2026-09-02T00:00:00Z")
    except ValueError:
        return
    raise AssertionError("future availability was not rejected")
