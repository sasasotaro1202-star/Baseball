from baseball_game_type import (
    classify_game,
    training_categories,
    evaluation_categories,
)


def test_npb_taxonomy_is_separate_and_defaults_are_fail_closed_for_specials():
    assert classify_game("NPB", game_type="公式戦")["category"] == "regular"
    assert classify_game("NPB", game_type="交流戦")["category"] == "interleague"
    assert classify_game("NPB", game_type="クライマックスシリーズ")["category"] == "climax"
    assert classify_game("NPB", game_type="日本シリーズ")["category"] == "japan_series"
    assert classify_game("NPB", game_type="オールスター")["category"] == "allstar"
    assert classify_game("NPB", game_type="親善試合")["category"] == "special"
    assert "climax" not in training_categories("NPB")
    assert "climax" in evaluation_categories("NPB")


def test_mlb_statsapi_game_types_are_separated():
    assert classify_game("MLB", game_type_code="R")["category"] == "regular"
    assert classify_game("MLB", game_type_code="F")["category"] == "wild_card"
    assert classify_game("MLB", game_type_code="D")["category"] == "division_series"
    assert classify_game("MLB", game_type_code="L")["category"] == "league_championship_series"
    assert classify_game("MLB", game_type_code="W")["category"] == "world_series"
    assert classify_game("MLB", game_type_code="A")["category"] == "allstar"
    assert classify_game("MLB", game_type_code="S")["category"] == "spring_training"
    assert "regular" in training_categories("MLB")
    assert "world_series" not in training_categories("MLB")
    assert "world_series" in evaluation_categories("MLB")


def test_international_taxonomy_includes_asian_games_without_mixing_into_club_training():
    result = classify_game("INTERNATIONAL", competition="20th Asian Games")
    assert result["category"] == "asian_games"
    assert result["training_default"] is False
    assert result["evaluation_default"] is True
    assert "asian_games" not in training_categories("INTERNATIONAL")
    assert "asian_games" in evaluation_categories("INTERNATIONAL")


def test_unknown_is_fail_closed():
    result = classify_game("MLB", game_type_code="???", competition="")
    assert result["category"] == "unknown"
    assert result["recognized"] is False
    assert result["training_default"] is False
    assert result["evaluation_default"] is False
