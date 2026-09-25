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


def test_recovery_cancels_legacy_acquisition_when_parallel_lane_is_active():
    text = Path("monitoring/recovery_controller.py").read_text(encoding="utf-8")
    assert "parallel_acquisition_active" in text
    assert "legacy_acquisition_superseded" in text
    assert 'row.get("_workflow_file") == "baseball-data-acquisition.yml"' in text
    assert 'row.get("_workflow_file") == "baseball-parallel-source-acquisition.yml"' in text
