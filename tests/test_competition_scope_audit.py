from pathlib import Path


def test_competition_audit_is_prediction_neutral():
    text = Path("research/competition_scope_audit.py").read_text(encoding="utf-8")
    assert '"research_only": True' in text
    assert '"prediction_affecting": False' in text
    assert "Unknown or missing metadata stays UNKNOWN/DEFERRED" in text


def test_competition_audit_tracks_npb_tournament_classes():
    text = Path("research/competition_scope_audit.py").read_text(encoding="utf-8")
    for category in ("climax", "japan_series", "allstar", "special"):
        assert f'"{category}"' in text


def test_mlb_acquisition_captures_official_game_type():
    text = Path("mlb_incremental_acquire.py").read_text(encoding="utf-8")
    assert '"game_type": str(game.get("gameType") or "").strip()' in text
    assert '"series_description": str(game.get("seriesDescription") or "").strip()' in text
