from monitoring.recovery_controller import stale_production_runs

def test_active_production_run_with_old_sha_is_stale():
    rows = [{"id": 811, "status": "in_progress", "head_sha": "old"}]
    assert stale_production_runs(rows, "new") == [811]

def test_active_production_run_with_current_sha_is_not_stale():
    rows = [{"id": 812, "status": "in_progress", "head_sha": "new"}]
    assert stale_production_runs(rows, "new") == []

def test_missing_sha_is_not_cancelled():
    rows = [{"id": 813, "status": "in_progress", "head_sha": ""}]
    assert stale_production_runs(rows, "new") == []
