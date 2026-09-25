from pathlib import Path

import pandas as pd

from baseball_backtest import BaseballBacktest


def test_npb_raw_schedule_fallback_loads_and_maps_competition_types(tmp_path):
    npb_dir = tmp_path / "npb"
    npb_dir.mkdir()
    frame = pd.DataFrame(
        [
            {
                "season": 2024,
                "ID": 1,
                "SeqNo": 1,
                "game_id": "regular-1",
                "game_kind_id": 1,
                "game_date": "2024-03-29 18:00:00+09:00",
                "game_state": 2,
                "home_score": 3,
                "away_score": 1,
                "home_team_short_name": "巨人",
                "away_team_short_name": "阪神",
                "stadium_name_jpn": "東京ドーム",
            },
            {
                "season": 2024,
                "ID": 2,
                "SeqNo": 1,
                "game_id": "inter-1",
                "game_kind_id": 26,
                "game_date": "2024-06-04 18:00:00+09:00",
                "game_state": 2,
                "home_score": 2,
                "away_score": 2,
                "home_team_short_name": "巨人",
                "away_team_short_name": "楽天",
                "stadium_name_jpn": "東京ドーム",
            },
            {
                "season": 2024,
                "ID": 3,
                "SeqNo": 1,
                "game_id": "allstar-1",
                "game_kind_id": 4,
                "game_date": "2024-07-23 18:30:00+09:00",
                "game_state": 2,
                "home_score": 6,
                "away_score": 8,
                "home_team_short_name": "全パ",
                "away_team_short_name": "全セ",
                "stadium_name_jpn": "エスコンＦ",
            },
            {
                "season": 2024,
                "ID": 4,
                "SeqNo": 1,
                "game_id": "cancelled-1",
                "game_kind_id": 1,
                "game_date": "2024-04-01 18:00:00+09:00",
                "game_state": 4,
                "home_score": 0,
                "away_score": 0,
                "home_team_short_name": "巨人",
                "away_team_short_name": "阪神",
                "stadium_name_jpn": "東京ドーム",
            },
        ]
    )
    frame.to_csv(npb_dir / "npb_games_2024_RAW_UNNORMALIZED.csv", index=False)

    bt = BaseballBacktest(data_dir=tmp_path)
    loaded = bt.load_npb_pbp()

    assert set(loaded["game_id"]) == {"regular-1", "inter-1", "allstar-1"}
    mapped = dict(zip(loaded["game_id"], loaded["game_type"]))
    assert mapped["regular-1"] == "regular season"
    assert mapped["inter-1"] == "interleague"
    assert mapped["allstar-1"] == "all-star"
    assert loaded["date"].dt.tz is not None
    assert any(x.get("type") == "npb_raw_fallback_used" for x in bt.audit)


def test_npb_raw_fallback_preserves_draws_and_uses_stable_game_identity(tmp_path):
    npb_dir = tmp_path / "npb"
    npb_dir.mkdir()
    frame = pd.DataFrame(
        [
            {
                "SeqNo": 1,
                "game_id": "g1",
                "game_kind_id": 1,
                "game_date": "2023-09-01 18:00:00+09:00",
                "game_state": 2,
                "home_score": 1,
                "away_score": 1,
                "home_team_short_name": "巨人",
                "away_team_short_name": "広島",
                "stadium_name_jpn": "東京ドーム",
            },
            {
                "SeqNo": 2,
                "game_id": "g1",
                "game_kind_id": 1,
                "game_date": "2023-09-01 18:00:00+09:00",
                "game_state": 2,
                "home_score": 1,
                "away_score": 1,
                "home_team_short_name": "巨人",
                "away_team_short_name": "広島",
                "stadium_name_jpn": "東京ドーム",
            },
        ]
    )
    frame.to_csv(npb_dir / "npb_games_2023_RAW_UNNORMALIZED.csv", index=False)

    bt = BaseballBacktest(data_dir=tmp_path)
    loaded = bt.aggregate_npb_games(bt.load_npb_pbp())

    assert len(loaded) == 1
    assert loaded.iloc[0]["home_score"] == 1
    assert loaded.iloc[0]["away_score"] == 1
    assert loaded.iloc[0]["npb_game_category"] == "regular"


def test_npb_raw_fallback_reads_tracked_repository_snapshots():
    data_dir = Path("data")
    raw_dir = data_dir / "npb"
    files = sorted(raw_dir.glob("npb_games_*_RAW_UNNORMALIZED.csv"))
    assert len(files) >= 8

    bt = BaseballBacktest(data_dir=data_dir)
    loaded = bt.load_npb_pbp()

    assert len(loaded) >= 5000
    years = pd.to_datetime(loaded["date"], utc=True).dt.year
    assert years.min() <= 2018
    assert years.max() >= 2025
    assert {"regular season", "interleague"}.issubset(set(loaded["game_type"].astype(str)))
    assert loaded["game_id"].nunique() == len(loaded)
