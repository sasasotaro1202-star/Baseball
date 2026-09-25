from pathlib import Path

def test_backtest_final_gate_checks_runtime_over_budget():
    text = Path("baseball_backtest.py").read_text()
    assert "runtime_over_budget = bool(runtime_seconds > self.time_budget_sec)" in text
    assert "if failures or budget_exhausted or runtime_over_budget:" in text
