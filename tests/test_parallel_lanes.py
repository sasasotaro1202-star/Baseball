from pathlib import Path


def test_acquisition_workflow_has_independent_npb_mlb_competition_jobs():
    text = Path(".github/workflows/baseball-data-acquisition.yml").read_text(encoding="utf-8")
    assert "  npb:" in text
    assert "  mlb:" in text
    assert "  competition:" in text
    assert "name: NPB historical + postseason acquisition" in text
    assert "name: MLB historical + postseason-ready acquisition" in text
    assert "name: Competition / tournament scope audit" in text


def test_competition_lane_uses_package_module_execution():
    text = Path(".github/workflows/baseball-data-acquisition.yml").read_text(encoding="utf-8")
    assert "python -m research.competition_scope_audit" in text
    assert "python research/competition_scope_audit.py |" not in text



def test_mlb_acquisition_does_not_depend_on_unrelated_production_results():
    text = Path(".github/workflows/baseball-mlb-competition.yml").read_text(encoding="utf-8")
    assert "python production_data_quality_gate.py" not in text
    assert "MLB cache unexpectedly small" in text
    assert "game_type metadata" in text
