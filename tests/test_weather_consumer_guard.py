import pandas as pd
from pathlib import Path

from baseball_backtest import BaseballBacktest


def test_weather_consumer_rejects_retired_inferred_availability():
    bt = BaseballBacktest()
    row = pd.Series({
        "prediction_time_utc": "2026-09-25T12:00:00Z",
        "weather_available_at": "2026-09-25T04:00:00Z",
        "weather_state": "PROJECTED",
        "weather_pit_quality": "CONSERVATIVE_8H_BOUND",
    })
    assert not bt._context_pit_safe(
        row,
        pd.Timestamp("2026-09-25T12:00:00Z"),
        "weather_available_at",
        "weather_state",
    )


def test_unknown_weather_provenance_fails_closed():
    bt = BaseballBacktest()
    row = pd.Series({
        "prediction_time_utc": "2026-09-25T12:00:00Z",
        "weather_available_at": "2026-09-25T04:00:00Z",
        "weather_state": "PROJECTED",
        "weather_pit_quality": "UNKNOWN",
    })
    assert not bt._context_pit_safe(
        row,
        pd.Timestamp("2026-09-25T12:00:00Z"),
        "weather_available_at",
        "weather_state",
    )
