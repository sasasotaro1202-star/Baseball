from pathlib import Path


def test_context_readiness_does_not_scan_future_rows():
    text = Path("baseball_backtest.py").read_text(encoding="utf-8")
    block_start = text.index("# Context readiness is learned strictly from rows already observed")
    block_end = text.index("    def update_after_game", block_start)
    block = text[block_start:block_end]
    assert "games.sort_values([\"datetime\", \"game_id\"])" in block
    assert "starter_valid_seen = starter_ok_seen = 0" in block
    assert "lineup_valid_seen = lineup_ok_seen = 0" in block
    assert "Update PIT support counters only after the current row is processed." in block
    assert "historical training corpus contains >=50%" not in block


def test_context_readiness_uses_only_training_rows():
    text = Path("baseball_backtest.py").read_text(encoding="utf-8")
    block_start = text.index("# Context readiness is learned strictly from rows already observed")
    block_end = text.index("    def update_after_game", block_start)
    block = text[block_start:block_end]
    assert "if train_include:" in block
    assert "starter_valid, starter_safe = _pit_observation_safe(row, \"starter\")" in block
    assert "lineup_valid, lineup_safe = _pit_observation_safe(row, \"lineup\")" in block
