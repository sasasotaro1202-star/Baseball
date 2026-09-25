from monitoring.recovery_controller import duplicate_production_run_ids

def test_duplicate_production_same_sha_only_cancels_queued():
    rows = [
        {"id": 10, "status": "in_progress", "head_sha": "abc"},
        {"id": 11, "status": "queued", "head_sha": "abc"},
        {"id": 12, "status": "queued", "head_sha": "def"},
        {"id": 13, "status": "in_progress", "head_sha": "def"},
    ]
    assert duplicate_production_run_ids(rows) == [11]

def test_duplicate_production_does_not_cancel_single_run():
    rows = [
        {"id": 10, "status": "in_progress", "head_sha": "abc"},
        {"id": 12, "status": "queued", "head_sha": "def"},
    ]
    assert duplicate_production_run_ids(rows) == []
