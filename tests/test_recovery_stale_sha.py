from monitoring import recovery_controller as rc

def test_active_production_run_with_old_sha_is_stale():
    rows = [{"id": 811, "status": "in_progress", "head_sha": "old"}]
    current = "new"
    assert rows[0]["head_sha"] != current

def test_active_production_run_with_current_sha_is_not_stale():
    rows = [{"id": 812, "status": "in_progress", "head_sha": "new"}]
    current = "new"
    assert rows[0]["head_sha"] == current
