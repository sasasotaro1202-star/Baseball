import pandas as pd

from baseball_backtest import BaseballBacktest


def test_recovered_regular_category_recomputes_training_and_evaluation_flags(tmp_path):
    raw_dir = tmp_path / "npb"
    src_dir = tmp_path / "npb_games"
    raw_dir.mkdir()
    src_dir.mkdir()

    pd.DataFrame([{
        "game_id": "g1",
        "game_kind_id": 1,
        "game_state": 2,
        "game_date": "2025-04-01 18:00:00+09:00",
        "home_team_short_name": "巨人",
        "away_team_short_name": "阪神",
        "home_score": 4,
        "away_score": 2,
        "stadium_name_jpn": "東京ドーム",
    }]).to_csv(raw_dir / "npb_games_2025_RAW_UNNORMALIZED.csv", index=False)

    pd.DataFrame([{
        "game_id": "g1",
        "datetime": "2025-04-01 18:00:00+09:00",
        "home": "巨人",
        "away": "阪神",
        "home_score": 4,
        "away_score": 2,
        "game_type": "UNKNOWN",
        "venue": "東京ドーム",
        "npb_game_category": "unknown",
        "npb_training_default": False,
        "npb_evaluation_default": False,
    }]).to_csv(src_dir / "2025_multi_source_pbp.csv", index=False)

    bt = BaseballBacktest(data_dir=tmp_path)
    out = bt.load_npb_pbp()
    row = out.iloc[0]
    assert row["npb_game_category"] == "regular"
    assert bool(row["npb_training_default"]) is True
    assert bool(row["npb_evaluation_default"]) is True


def test_recovered_excluded_category_remains_ineligible(tmp_path):
    raw_dir = tmp_path / "npb"
    src_dir = tmp_path / "npb_games"
    raw_dir.mkdir()
    src_dir.mkdir()

    pd.DataFrame([{
        "game_id": "g2",
        "game_kind_id": 5,
        "game_state": 2,
        "game_date": "2025-03-10 18:00:00+09:00",
        "home_team_short_name": "巨人",
        "away_team_short_name": "阪神",
        "home_score": 3,
        "away_score": 2,
        "stadium_name_jpn": "東京ドーム",
    }]).to_csv(raw_dir / "npb_games_2025_RAW_UNNORMALIZED.csv", index=False)

    pd.DataFrame([{
        "game_id": "g2",
        "datetime": "2025-03-10 18:00:00+09:00",
        "home": "巨人",
        "away": "阪神",
        "home_score": 3,
        "away_score": 2,
        "game_type": "UNKNOWN",
        "venue": "東京ドーム",
        "npb_game_category": "unknown",
        "npb_training_default": False,
        "npb_evaluation_default": False,
    }]).to_csv(src_dir / "2025_multi_source_pbp.csv", index=False)

    bt = BaseballBacktest(data_dir=tmp_path)
    out = bt.load_npb_pbp()
    # This loader preserves rows for audit; downstream aggregation excludes them.
    assert out.loc[0, "npb_game_category"] == "excluded"
    assert bool(out.loc[0, "npb_training_default"]) is False
    assert bool(out.loc[0, "npb_evaluation_default"]) is False
