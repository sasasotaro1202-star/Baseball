from pathlib import Path


def test_acquisition_workflow_is_an_independent_npb_lane():
    text = Path(".github/workflows/baseball-parallel-source-acquisition.yml").read_text(encoding="utf-8")
    assert "  npb:" in text
    assert "NPB full-scope resumable acquisition" in text
    assert "continue-on-error: true" in text
    assert "  mlb:" not in text
    assert "  competition:" not in text


def test_competition_lane_uses_package_module_execution():
    text = Path(".github/workflows/baseball-mlb-competition.yml").read_text(encoding="utf-8")
    assert "  competition:" in text
    assert "python -m research.competition_scope_audit" in text
    assert "python research/competition_scope_audit.py |" not in text


def test_mlb_acquisition_does_not_depend_on_unrelated_production_results():
    text = Path(".github/workflows/baseball-mlb-competition.yml").read_text(encoding="utf-8")
    assert "python production_data_quality_gate.py" not in text
    assert "MLB cache unexpectedly small" in text
    assert "game_type metadata" in text


def test_competition_and_matchday_are_separate_workflows():
    for path in (
        ".github/workflows/baseball-mlb-competition.yml",
        ".github/workflows/baseball_matchday.yml",
        ".github/workflows/baseball_research.yml",
        ".github/workflows/baseball_production.yml",
    ):
        assert Path(path).exists()
