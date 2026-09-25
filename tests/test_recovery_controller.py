from pathlib import Path


def test_recovery_controller_is_syntax_safe_and_tracks_parallel_lanes():
    text = Path("monitoring/recovery_controller.py").read_text(encoding="utf-8")
    compile(text, "monitoring/recovery_controller.py", "exec")
    assert "baseball-parallel-source-acquisition.yml" in text
    assert "baseball-mlb-competition.yml" in text
    assert "ACTIVE_STATES" in text
    assert "recently_created" in text


def test_recovery_controller_treats_legacy_and_new_acquisition_as_one_logical_lane():
    text = Path("monitoring/recovery_controller.py").read_text(encoding="utf-8")
    assert '"acquisition": [' in text
    assert '"baseball-parallel-source-acquisition.yml"' in text
    assert '"baseball-data-acquisition.yml"' in text
