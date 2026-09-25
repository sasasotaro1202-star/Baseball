from datetime import datetime, timezone
import pandas as pd

from production_matchday_intelligence import rest_travel


def test_rest_travel_ignores_future_games():
    historical = pd.DataFrame([
        {"home":"A","away":"B","datetime":"2026-09-24T18:00:00Z","venue":"東京ドーム"},
        {"home":"A","away":"C","datetime":"2026-09-26T18:00:00Z","venue":"横浜"},
    ])
    game = {
        "datetime": pd.Timestamp("2026-09-25T09:00:00Z"),
        "home":"A","away":"B","venue":"東京ドーム",
    }
    out = rest_travel(historical, game)
    assert out["state"] == "VERIFIED"
    assert out["home"]["history_cutoff"] == "strictly_before_target"
    assert out["home"]["rest_days"] > 0
    assert out["home"]["travel_miles"] == 0.0


def test_rest_travel_uses_target_window_not_previous_game_relative_window():
    historical = pd.DataFrame([
        {"home":"A","away":"B","datetime":"2026-09-20T18:00:00Z","venue":"東京ドーム"},
        {"home":"A","away":"C","datetime":"2026-09-23T18:00:00Z","venue":"東京ドーム"},
    ])
    game = {
        "datetime": pd.Timestamp("2026-09-25T18:00:00Z"),
        "home":"A","away":"B","venue":"東京ドーム",
    }
    out = rest_travel(historical, game)
    assert out["home"]["games_last_3d"] == 1
    assert out["home"]["games_last_7d"] == 2



def test_rest_travel_missing_side_is_unknown_not_fabricated():
    historical = pd.DataFrame([
        {"home":"A","away":"C","datetime":"2026-09-24T18:00:00Z","venue":"東京ドーム"},
    ])
    game = {
        "datetime": pd.Timestamp("2026-09-25T18:00:00Z"),
        "home":"A","away":"B","venue":"東京ドーム",
    }
    out = rest_travel(historical, game)
    assert out["home"]["state"] == "VERIFIED"
    assert out["away"]["state"] == "UNKNOWN"
    assert out["state"] == "PARTIAL"
