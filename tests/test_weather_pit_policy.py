from pathlib import Path

def test_historical_weather_does_not_infer_pit_availability():
    text = Path("npb_multi_source.py").read_text(encoding="utf-8")
    assert "weather_available_at']=''" in text
    assert "FAIL_CLOSED_NO_ISSUANCE_TIMESTAMP" in text
    assert "valid-pd.Timedelta(hours=8)" not in text

def test_legacy_weather_cache_is_sanitized_before_replay():
    text = Path("npb_multi_source.py").read_text(encoding="utf-8")
    assert "Sanitize any legacy cache" in text
    assert "cache['weather_available_at']=''" in text
    assert "cache['weather_state']='UNKNOWN'" in text
