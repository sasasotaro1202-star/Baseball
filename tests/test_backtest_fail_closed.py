from pathlib import Path


def test_walkforward_model_failure_is_not_silently_skipped():
    text = Path("baseball_backtest.py").read_text(encoding="utf-8")
    assert '"type": "model_block_failure"' in text
    assert 'FAIL-CLOSED' in text
    assert 'raise RuntimeError(' in text


def test_checkpoint_write_failure_is_not_silently_ignored():
    text = Path("baseball_backtest.py").read_text(encoding="utf-8")
    assert '"type":"checkpoint_write_error"' in text
    assert 'checkpoint write failed' in text


def test_run_marks_runtime_over_budget_as_failure():
    text = Path("baseball_backtest.py").read_text(encoding="utf-8")
    assert 'runtime_seconds > self.time_budget_sec' in text
    assert '"status": "FAIL"' in text
