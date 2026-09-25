from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from baseball_backtest import BaseballBacktest


EXPECTED_YEARS = tuple(range(2018, 2026))
REQUIRED_COLUMNS = {
    "game_id",
    "game_date",
    "home_team_short_name",
    "away_team_short_name",
    "home_score",
    "away_score",
    "stadium_name_jpn",
    "game_kind_id",
    "game_state",
}


def test_tracked_npb_raw_fallback_is_structurally_sound(tmp_path: Path) -> None:
    source_dir = Path("data/npb")
    target_dir = tmp_path / "data" / "npb"
    target_dir.mkdir(parents=True)

    frames = []
    for year in EXPECTED_YEARS:
        src = source_dir / f"npb_games_{year}_RAW_UNNORMALIZED.csv"
        assert src.exists(), f"missing tracked NPB fallback snapshot: {src}"
        frame = pd.read_csv(src, usecols=lambda c: c in REQUIRED_COLUMNS or c == "SeqNo")
        assert REQUIRED_COLUMNS.issubset(frame.columns), f"missing columns in {src.name}"
        assert frame["game_id"].notna().all(), f"null game_id in {src.name}"
        assert frame["game_state"].eq(2).any(), f"no completed rows in {src.name}"
        frames.append(frame)
        shutil.copy2(src, target_dir / src.name)

    raw = pd.concat(frames, ignore_index=True)
    completed = raw.loc[
        raw["game_state"].eq(2)
        & raw["home_score"].notna()
        & raw["away_score"].notna()
    ].copy()

    assert len(completed) >= 7_000
    assert completed["game_id"].astype(str).is_unique

    dates = pd.to_datetime(completed["game_date"], errors="coerce", utc=True)
    assert dates.notna().all()
    assert dates.min().year == 2018
    assert dates.max().year == 2025

    kinds = set(pd.to_numeric(completed["game_kind_id"], errors="coerce").dropna().astype(int))
    assert {1, 2, 11} & kinds, "no regular-season game type present"
    assert 26 in kinds, "no interleague game type present"

    bt = BaseballBacktest(data_dir=tmp_path / "data")
    loaded = bt.load_npb_pbp()

    assert len(loaded) == completed["game_id"].astype(str).nunique()
    assert loaded["game_id"].astype(str).is_unique
    assert loaded["date"].notna().all()
    assert loaded["home"].notna().all()
    assert loaded["away"].notna().all()
    assert loaded["home_pitcher"].eq("").all()
    assert loaded["away_pitcher"].eq("").all()
