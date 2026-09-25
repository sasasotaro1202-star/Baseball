from pathlib import Path


def test_integrity_gate_checks_prediction_probability_columns():
    text = Path("production_integrity_gate.py").read_text(encoding="utf-8")
    for marker in ("pred_home", "pred_draw", "pred_away"):
        assert marker in text
    assert "probability out of range" in text
    assert "probability mass does not sum to 1" in text


def test_matchday_validator_cannot_pass_with_deferred_predictions():
    text = Path(".github/workflows/baseball_matchday.yml").read_text(encoding="utf-8")
    assert "if x.get('prediction_status') == 'PASS':" in text
    assert "elif x.get('prediction_status') == 'DEFERRED':" in text
    assert "assert x.get('status') == 'DEFERRED'" in text
