from pathlib import Path

def test_production_budget_cap_matches_workflow():
    text = Path("baseball_backtest.py").read_text()
    assert 'float(os.getenv("BASEBALL_TIME_BUDGET_SEC", "3600"))' in text
    assert '3600.0' in text
