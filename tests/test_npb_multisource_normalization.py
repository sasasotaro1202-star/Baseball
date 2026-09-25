import pandas as pd

from baseball_backtest import BaseballBacktest


def test_multisource_loader_normalizes_collector_schema(tmp_path):
    d = tmp_path / "npb_games"
    d.mkdir()
    pd.DataFrame([{
        "game_id":"g1",
        "datetime":"2025-04-01 18:00:00+09:00",
        "home":"巨人",
        "away":"阪神",
        "home_score":4,
        "away_score":2,
        "game_type":"regular season",
        "venue":"東京ドーム",
        "home_starter":"H",
        "away_starter":"A",
        "home_pregame_starter":"",
        "away_pregame_starter":"",
        "starter_available_at":"",
        "starter_state":"UNKNOWN",
        "weather_available_at":"",
        "weather_state":"UNKNOWN",
    }]).to_csv(d / "2025_multi_source_pbp.csv", index=False)

    bt = BaseballBacktest(data_dir=tmp_path)
    pbp = bt.load_npb_pbp()
    assert pbp["game_id"].tolist() == ["g1"]
    assert str(pbp["date"].dt.tz) == "UTC"
    assert pbp.loc[0, "home_pitcher"] == "H"
    assert pbp.loc[0, "away_pitcher"] == "A"
    games = bt.aggregate_npb_games(pbp)
    assert len(games) == 1
    assert bool(games.loc[0, "confirmed_starters"])


def test_multisource_loader_fails_closed_without_valid_rows(tmp_path):
    d = tmp_path / "npb_games"
    d.mkdir()
    pd.DataFrame([{
        "game_id":"bad",
        "datetime":"not-a-date",
        "home":"A",
        "away":"B",
        "home_score":1,
        "away_score":0,
    }]).to_csv(d / "2025_multi_source_pbp.csv", index=False)

    bt = BaseballBacktest(data_dir=tmp_path)
    try:
        bt.load_npb_pbp()
    except RuntimeError as exc:
        assert "No valid NPB rows" in str(exc)
    else:
        raise AssertionError("invalid multi-source rows must fail closed")


def test_tracked_multisource_file_has_expected_normalized_columns():
    source = "data/npb_games/2019_multi_source_pbp.csv"
    frame = pd.read_csv(source, nrows=1)
    assert {"game_id","datetime","home","away","home_score","away_score","home_pregame_starter","away_pregame_starter"} <= set(frame.columns)
