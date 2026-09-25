from pathlib import Path

def test_matchday_ensemble_proba_passes_league():
    text = Path("production_matchday_intelligence.py").read_text()
    assert 'bt.ensemble_proba(fitted,fx,"NPB")[0]' in text
