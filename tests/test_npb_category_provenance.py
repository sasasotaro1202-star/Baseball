import pandas as pd

from baseball_backtest import BaseballBacktest


def test_aggregate_preserves_stored_npb_category():
    bt = BaseballBacktest(data_dir=__import__("pathlib").Path("."))
    frame = pd.DataFrame([{
        "game_id":"g1","row_order":0,"home":"巨人","away":"阪神",
        "date":"2025-04-01T09:00:00+00:00","home_score":4,"away_score":2,
        "venue":"東京ドーム","game_type":"UNKNOWN",
        "home_pitcher":"H","away_pitcher":"A",
        "npb_game_category":"regular","npb_type_confidence":"high",
        "npb_training_default":True,"npb_evaluation_default":True,
    }])
    out = bt.aggregate_npb_games(frame)
    assert len(out) == 1
    assert out.loc[0, "npb_game_category"] == "regular"


def test_multisource_normalization_recovers_category_from_raw_fallback(tmp_path):
    npb_games = tmp_path / "npb_games"
    npb_games.mkdir()
    raw_dir = tmp_path / "npb"
    raw_dir.mkdir()
    pd.DataFrame([{
        "game_id":"g1","game_kind_id":1,"game_state":2,
        "game_date":"2025-04-01 18:00:00+09:00",
        "home_team_short_name":"巨人","away_team_short_name":"阪神",
        "home_score":4,"away_score":2,"stadium_name_jpn":"東京ドーム",
    }]).to_csv(raw_dir / "npb_games_2025_RAW_UNNORMALIZED.csv", index=False)
    pd.DataFrame([{
        "game_id":"g1","datetime":"2025-04-01 18:00:00+09:00",
        "home":"巨人","away":"阪神","home_score":4,"away_score":2,
        "game_type":"UNKNOWN","venue":"東京ドーム",
    }]).to_csv(npb_games / "2025_multi_source_pbp.csv", index=False)

    bt = BaseballBacktest(data_dir=tmp_path)
    pbp = bt.load_npb_pbp()
    assert pbp.loc[0, "npb_game_category"] == "regular"
    assert bool(pbp.loc[0, "npb_training_default"]) is True
