from pathlib import Path


def test_acquisition_keeps_mlb_independent_from_npb_completion():
    text = Path(".github/workflows/baseball-data-acquisition.yml").read_text(encoding="utf-8")
    assert "continue-on-error: true" in text
    assert "name: MLB full-scope acquisition with incremental refresh" in text
    assert "if: steps.npb_gate.outputs.complete == 'true'" not in text
    assert "name: Persist MLB readiness checkpoint" in text
    assert "name: Finalize NPB acquisition outcome" in text
