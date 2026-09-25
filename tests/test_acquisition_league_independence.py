from pathlib import Path


def test_acquisition_keeps_mlb_independent_from_npb_completion():
    text = Path(".github/workflows/baseball-mlb-competition.yml").read_text(encoding="utf-8")
    assert "continue-on-error: true" in text
    assert "name: MLB full-scope acquisition with incremental refresh" in text
    assert "if: steps.npb" not in text
    assert "Baseball Parallel NPB Source Acquisition" not in text
    assert "Validate MLB cache" in text
