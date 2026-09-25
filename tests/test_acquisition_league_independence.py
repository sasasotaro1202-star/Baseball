from pathlib import Path


def test_acquisition_uses_independent_npb_lane():
    text = Path(".github/workflows/baseball-parallel-source-acquisition.yml").read_text(encoding="utf-8")
    assert "  npb:" in text
    assert "name: NPB full-scope resumable acquisition" in text
    assert "continue-on-error: true" in text


def test_mlb_acquisition_is_not_blocked_by_npb_lane():
    text = Path(".github/workflows/baseball-mlb-competition.yml").read_text(encoding="utf-8")
    assert "  mlb:" in text
    assert "MLB historical + postseason acquisition" in text
    assert "  competition:" in text
    assert "npb_collection_status" not in text
