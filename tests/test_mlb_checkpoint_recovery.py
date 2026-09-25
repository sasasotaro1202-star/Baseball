from pathlib import Path


def test_mlb_checkpoint_recovery_handles_empty_data_error():
    text = Path("mlb_incremental_acquire.py").read_text(encoding="utf-8")
    assert "except pd.errors.EmptyDataError" in text
    assert 'reason = "empty-data checkpoint"' in text
    assert "quarantine" in text


def test_mlb_checkpoint_refreshes_stale_schema_with_game_type():
    text = Path("mlb_incremental_acquire.py").read_text(encoding="utf-8")
    assert '"game_type": str(game.get("gameType") or "").strip()' in text
    assert "checkpoint lacks official game_type" in text
