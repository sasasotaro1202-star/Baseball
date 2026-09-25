from pathlib import Path

def test_historical_weather_does_not_infer_pit_availability():
    text = Path("npb_multi_source.py").read_text(encoding="utf-8")
    assert "weather_available_at']=''" in text
    assert "FAIL_CLOSED_NO_ISSUANCE_TIMESTAMP" in text
    assert "valid-pd.Timedelta(hours=8)" not in text
