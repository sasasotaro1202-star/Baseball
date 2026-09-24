#!/usr/bin/env python3
"""Deterministic Baseball CI runtime smoke tests."""
from __future__ import annotations

import pandas as pd

from baseball_backtest import BaseballBacktest


def main() -> int:
    b = BaseballBacktest()
    row = pd.Series({
        "league": "NPB",
        "datetime": pd.Timestamp("2026-09-24T18:00:00"),
        "home": "広島東洋カープ",
        "away": "読売ジャイアンツ",
        "home_starter": "A",
        "away_starter": "B",
        "home_lineup_json": "[]",
        "away_lineup_json": "[]",
        "weather_temp_c": 25.0,
        "weather_humidity_pct": 60.0,
        "weather_wind_kmh": 8.0,
        "weather_precip_mm": 0.0,
    })

    first = b.match_features(row)
    assert "h_games_last_3d" in first
    assert "home_bullpen_short_rest" in first
    assert first["weather_pit_safe"] == 0.0
    assert first["weather_temp_c"] == 0.0

    safe_row = row.copy()
    safe_row["prediction_time_utc"] = "2026-09-24T08:00:00Z"
    safe_row["weather_available_at"] = "2026-09-24T07:00:00Z"
    safe_row["weather_state"] = "PROJECTED"
    safe_row["lineup_available_at"] = "2026-09-24T07:00:00Z"
    safe_row["lineup_state"] = "PROJECTED"
    safe = b.match_features(safe_row)
    assert safe["weather_pit_safe"] != 0.0
    assert safe["weather_temp_c"] == 25.0

    sh = b.state("NPB", "広島東洋カープ")
    sa = b.state("NPB", "読売ジャイアンツ")
    b._update_team(sh, 0, 5, 2, True, 3, pd.Timestamp("2026-09-23T18:00:00"), row, opponent="読売ジャイアンツ")
    b._update_team(sa, 2, 2, 5, False, 0, pd.Timestamp("2026-09-23T18:00:00"), row, opponent="広島東洋カープ")
    later = b._team_features(
        "NPB",
        "広島東洋カープ",
        "home",
        pd.Timestamp("2026-09-24T18:00:00"),
        opponent="読売ジャイアンツ",
    )
    assert later["games_last_3d"] == 1.0
    assert later["same_opponent_last_10"] == 1.0
    print("Baseball core runtime smoke: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
