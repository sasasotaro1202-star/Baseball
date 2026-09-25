from pathlib import Path


def test_npb_coverage_gate_has_separate_critical_training_start():
    text = Path("npb_multi_source.py").read_text(encoding="utf-8")
    assert 'NPB_REQUIRED_START_YEAR' in text
    assert 'critical_start=max(START_YEAR' in text
    assert "critical training-era starter-line coverage below threshold" in text


def test_npb_coverage_gate_does_not_block_on_reference_only_seasons():
    text = Path("npb_multi_source.py").read_text(encoding="utf-8")
    assert "reference-only seasons" in text
    assert "they remain available but are not allowed to block readiness" in text
