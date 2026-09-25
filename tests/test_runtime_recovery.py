from pathlib import Path


def test_recovery_controller_compiles():
    text = Path("monitoring/recovery_controller.py").read_text(encoding="utf-8")
    compile(text, "monitoring/recovery_controller.py", "exec")


def test_recovery_tracks_parallel_lanes_without_elif_serialization():
    text = Path("monitoring/recovery_controller.py").read_text(encoding="utf-8")
    for lane in (
        '"validation"',
        '"production"',
        '"acquisition"',
        '"mlb_competition"',
        '"mac"',
        '"research"',
        '"matchday"',
        '"game_type_ablation"',
    ):
        assert lane in text
    assert "baseball-parallel-source-acquisition.yml" in text
    assert "baseball-mlb-competition.yml" in text
    assert "elif " not in text


def test_recovery_treats_legacy_and_parallel_acquisition_as_one_logical_lane():
    text = Path("monitoring/recovery_controller.py").read_text(encoding="utf-8")
    assert '"acquisition": [' in text
    assert '"baseball-parallel-source-acquisition.yml"' in text
    assert '"baseball-data-acquisition.yml"' in text


def test_recovery_has_recent_dispatch_guard():
    text = Path("monitoring/recovery_controller.py").read_text(encoding="utf-8")
    assert "RECENT_DISPATCH_GUARD_MINUTES" in text
    assert "recently_created" in text



def test_recovery_does_not_let_stale_validation_block_current_main():
    controller = Path(__file__).resolve().parents[1] / "monitoring" / "recovery_controller.py"
    text = controller.read_text(encoding="utf-8")
    assert "def active_current_sha" in text
    assert 'str(row.get("head_sha") or "") == str(sha or "")' in text
    assert 'dispatch("validation", "validate-code.yml")' in text
    assert "not active_current_sha(\"validation\", current_sha)" in text
