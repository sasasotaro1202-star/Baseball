from pathlib import Path
import pandas as pd

from npb_multi_source import resume_missing_game_ids


def test_light_mode_uses_core_checkpoint_for_resume() -> None:
    games = pd.DataFrame({"game_id": ["a", "b", "c"]})
    checkpoint = pd.DataFrame({"game_id": ["a", "b"]})
    assert resume_missing_game_ids(games, checkpoint, set(), True) == {"c"}


def test_full_mode_uses_player_enrichment_for_resume() -> None:
    games = pd.DataFrame({"game_id": ["a", "b", "c"]})
    checkpoint = pd.DataFrame({"game_id": ["a", "b"]})
    assert resume_missing_game_ids(games, checkpoint, {"a", "b"}, False) == {"c"}


def test_light_mode_does_not_require_player_rows() -> None:
    games = pd.DataFrame({"game_id": ["a", "b"]})
    checkpoint = pd.DataFrame({"game_id": ["a"]})
    # LIGHT mode intentionally has no player rows, so completed core game "a"
    # must remain completed across process restarts.
    assert resume_missing_game_ids(games, checkpoint, set(), True) == {"b"}
