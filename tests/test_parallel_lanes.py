from pathlib import Path


def test_acquisition_workflow_has_independent_npb_mlb_competition_jobs():
    npb = Path(".github/workflows/baseball-parallel-source-acquisition.yml").read_text(encoding="utf-8")
    mlb = Path(".github/workflows/baseball-mlb-competition.yml").read_text(encoding="utf-8")
    assert "  npb:" in npb
    assert "name: NPB full-scope resumable acquisition" in npb
    assert "  mlb:" in mlb
    assert "  competition:" in mlb
    assert "name: MLB historical + postseason acquisition" in mlb
    assert "name: MLB + cross-competition scope audit" in mlb


def test_competition_lane_uses_package_module_execution():
    text = Path(".github/workflows/baseball-mlb-competition.yml").read_text(encoding="utf-8")
    assert "python -m research.competition_scope_audit" in text
    assert "python research/competition_scope_audit.py |" not in text


def test_mlb_acquisition_does_not_depend_on_unrelated_production_results():
    text = Path(".github/workflows/baseball-mlb-competition.yml").read_text(encoding="utf-8")
    assert "python production_data_quality_gate.py" not in text
    assert "MLB cache unexpectedly small" in text
    assert "game_type metadata" in text
