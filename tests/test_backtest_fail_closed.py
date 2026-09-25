[object Object]


def test_model_cv_failures_are_auditable():
    text = Path("baseball_backtest.py").read_text(encoding="utf-8")
    assert '"type": "model_cv_error"' in text
    assert '"validation_size": int(val)' in text
