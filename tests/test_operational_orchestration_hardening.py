from pathlib import Path

def test_production_runtime_budget_is_fail_closed():
    text = Path("baseball_backtest.py").read_text(encoding="utf-8")
    assert "runtime_over_budget = bool(runtime_seconds > self.time_budget_sec)" in text
    assert "if failures or budget_exhausted or runtime_over_budget:" in text


def test_recovery_allows_only_matchday_snapshot_advances():
    text = Path("monitoring/recovery_controller.py").read_text(encoding="utf-8")
    assert 'NONSEMANTIC_PRODUCTION_PATH_PREFIXES = ("results/matchday_",)' in text
    assert "production_update_is_nonsemantic" in text
    assert "/compare/{row_sha}...{current_sha}" in text


def test_research_checkout_can_persist_state():
    text = Path(".github/workflows/baseball_research.yml").read_text(encoding="utf-8")
    assert "persist-credentials: true" in text


def test_validation_retriggers_on_production_data_updates():
    text = Path(".github/workflows/validate-code.yml").read_text(encoding="utf-8")
    for path in (
        "data/checkpoints/npb_collection_status.json",
        "data/mlb_games.csv",
        "data/checkpoints/mlb_collection_status.json",
    ):
        assert path in text


def test_runtime_budget_hit_is_fail_closed():
    text = Path("baseball_backtest.py").read_text(encoding="utf-8")
    assert "if time.time() - self.started_at >= self.time_budget_sec:" in text
    assert "budget_exhausted = True" in text


def test_stale_oos_artifact_preserves_backtest_results():
    text = Path(".github/workflows/baseball_production.yml").read_text(encoding="utf-8")
    assert "results/runtime_summary.csv" in text
    assert "results/*_backtest_results.csv" in text
    assert "results/checkpoints/**" in text
