from production_matchday_intelligence import prediction_eligibility


def test_matchday_prediction_requires_verified_starters():
    ok, reason = prediction_eligibility({"starter_state": "UNKNOWN"}, True)
    assert ok is False
    assert reason == "STARTERS_UNCONFIRMED"


def test_matchday_prediction_requires_model():
    ok, reason = prediction_eligibility({"starter_state": "VERIFIED"}, False)
    assert ok is False
    assert reason == "MODEL_UNAVAILABLE"


def test_matchday_prediction_ready_only_with_model_and_starters():
    ok, reason = prediction_eligibility({"starter_state": "VERIFIED"}, True)
    assert ok is True
    assert reason == "READY"


def test_mlb_probable_starter_is_not_confirmed_without_explicit_signal(monkeypatch):
    from baseball_backtest import BaseballBacktest

    class Stub(BaseballBacktest):
        def _get_json(self, *args, **kwargs):
            return {
                "dates": [{"games": [{
                    "gamePk": 123,
                    "gameDate": "2026-09-25T19:00:00Z",
                    "teams": {
                        "home": {"team": {"name": "Home"}, "probablePitcher": {"fullName": "Home P"}},
                        "away": {"team": {"name": "Away"}, "probablePitcher": {"fullName": "Away P"}},
                    },
                }]}]
            }

    obj = object.__new__(Stub)
    out = obj.current_mlb_schedule("2026-09-25")
    assert bool(out.iloc[0]["probable_starters"]) is True
    assert bool(out.iloc[0]["confirmed_starters"]) is False
    assert out.iloc[0]["starter_state"] == "PROJECTED"


def test_schedule_state_tracks_started_games_without_prediction_leakage():
    import pandas as pd
    from production_matchday_intelligence import classify_schedule_states

    now = pd.Timestamp("2026-09-25T18:30:00", tz="Asia/Tokyo")
    rows = classify_schedule_states([
        {"date":"2026-09-25","hour":18,"minute":0,"home":"A","away":"B","venue":"X"},
        {"date":"2026-09-26","hour":14,"minute":0,"home":"A","away":"C","venue":"Y"},
    ], now)

    assert rows[0]["schedule_state"] == "STARTED_OR_IN_PROGRESS"
    assert rows[1]["schedule_state"] == "FUTURE"
    assert rows[0]["home"] == "A"
    assert "score" not in rows[0]
    assert "actual" not in rows[0]
