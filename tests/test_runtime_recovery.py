from pathlib import Path


def test_recovery_uses_dedicated_controller():
    workflow = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "baseball_recovery.yml"
    text = workflow.read_text(encoding="utf-8")
    assert "python3 monitoring/recovery_controller.py" in text
    assert "Baseball Parallel NPB Source Acquisition" in text
    assert "Baseball MLB + Competition Research" in text
    assert "Baseball Matchday Intelligence" in text


def test_recovery_controller_defines_independent_lanes():
    controller = Path(__file__).resolve().parents[1] / "monitoring" / "recovery_controller.py"
    text = controller.read_text(encoding="utf-8")
    for label in (
        '"validation":', '"production":', '"acquisition":',
        '"mlb_competition":', '"mac":', '"research":',
        '"matchday":', '"game_type_ablation":',
    ):
        assert label in text
    assert '"baseball-parallel-source-acquisition.yml"' in text
    assert '"baseball-mlb-competition.yml"' in text
    assert '"baseball_matchday.yml"' in text


def test_recovery_dispatches_mlb_competition_when_needed():
    controller = Path(__file__).resolve().parents[1] / "monitoring" / "recovery_controller.py"
    text = controller.read_text(encoding="utf-8")
    assert "mlb_competition_failed" in text
    assert 'dispatch("mlb_competition", "baseball-mlb-competition.yml")' in text
    assert 'dispatch("matchday", "baseball_matchday.yml")' in text
    assert "recently_created" in text
