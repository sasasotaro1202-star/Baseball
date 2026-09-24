from pathlib import Path


def test_frozen_holdout_writes_to_repository_results():
    import research.frozen_holdout_gate as gate

    repo_root = Path(__file__).resolve().parents[1]
    assert gate.RESULTS == repo_root / "results"


def test_matchday_change_ledger_is_package_importable():
    import research.matchday_change_ledger as ledger

    assert ledger.RESULTS == Path("results")


def test_venue_effect_is_past_only():
    from baseball_backtest import BaseballBacktest
    import pandas as pd

    bt = BaseballBacktest()
    row = pd.Series({
        "league": "NPB", "home": "阪神タイガース", "away": "読売ジャイアンツ",
        "datetime": pd.Timestamp("2026-09-25T09:00:00Z"),
        "venue": "甲子園", "home_score": 5, "away_score": 2,
    })
    before = bt.match_features(row)["venue_known"]
    assert before == 0.0
    bt.update_after_game(row)
    after = bt.match_features(row)["venue_known"]
    assert after == 0.0  # one prior game is intentionally below the stability threshold
    for i in range(2):
        r = row.copy()
        r["datetime"] = pd.Timestamp("2026-09-26T09:00:00Z") + pd.Timedelta(days=i)
        r["home_score"] = 4 + i
        r["away_score"] = 1
        bt.update_after_game(r)
    target = row.copy()
    target["datetime"] = pd.Timestamp("2026-09-28T09:00:00Z")
    assert bt.match_features(target)["venue_known"] == 1.0

def test_common_venue_columns_are_preserved():
    from baseball_backtest import BaseballBacktest
    import pandas as pd

    bt = BaseballBacktest()
    g = pd.DataFrame({
        "game_id": ["1"], "home": ["阪神タイガース"], "away": ["読売ジャイアンツ"],
        "date": ["2026-09-01"], "game_type": ["公式戦"],
        "home_score": [3], "away_score": [1],
        "home_pitcher": ["home-sp"], "away_pitcher": ["away-sp"],
        "stadium_name": ["甲子園"], "row_order": [1],
    })
    out = bt.aggregate_npb_games(g)
    assert out.loc[0, "venue"] == "甲子園"
