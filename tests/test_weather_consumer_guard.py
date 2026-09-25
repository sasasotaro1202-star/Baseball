from pathlib import Path

def test_weather_consumer_rejects_retired_inferred_availability():
    text = Path("baseball_backtest.py").read_text(encoding="utf-8")
    assert "CONSERVATIVE_8H_BOUND" in text
    assert "weather_pit_quality" in text
    assert "only explicitly verified provenance may pass" in text
