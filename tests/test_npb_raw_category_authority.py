import pandas as pd
from baseball_backtest import BaseballBacktest


def test_raw_category_overrides_conflicting_stored_category(tmp_path):
    raw = tmp_path / "npb"; src = tmp_path / "npb_games"
    raw.mkdir(); src.mkdir()
    pd.DataFrame([{
        "game_id":"g1","game_kind_id":1,"game_state":2,
        "game_date":"2025-04-01 18:00:00+09:00",
        "home_team_short_name":"巨人","away_team_short_name":"阪神",
        "home_score":4,"away_score":2,"stadium_name_jpn":"東京ドーム",
    }]).to_csv(raw/"npb_games_2025_RAW_UNNORMALIZED.csv",index=False)
    pd.DataFrame([{
        "game_id":"g1","datetime":"2025-04-01 18:00:00+09:00",
        "home":"巨人","away":"阪神","home_score":4,"away_score":2,
        "game_type":"UNKNOWN","venue":"東京ドーム",
        "npb_game_category":"excluded","npb_training_default":False,
        "npb_evaluation_default":False,
    }]).to_csv(src/"2025_multi_source_pbp.csv",index=False)
    row=BaseballBacktest(data_dir=tmp_path).load_npb_pbp().iloc[0]
    assert row["npb_game_category"]=="regular"
    assert bool(row["npb_training_default"]) is True
    assert bool(row["npb_evaluation_default"]) is True


def test_stored_category_is_fallback_without_raw_overlap(tmp_path):
    src=tmp_path/"npb_games"; src.mkdir()
    pd.DataFrame([{
        "game_id":"g2","datetime":"2025-04-01 18:00:00+09:00",
        "home":"巨人","away":"阪神","home_score":2,"away_score":3,
        "game_type":"UNKNOWN","venue":"東京ドーム",
        "npb_game_category":"interleague","npb_training_default":True,
        "npb_evaluation_default":True,
    }]).to_csv(src/"2025_multi_source_pbp.csv",index=False)
    row=BaseballBacktest(data_dir=tmp_path).load_npb_pbp().iloc[0]
    assert row["npb_game_category"]=="interleague"
