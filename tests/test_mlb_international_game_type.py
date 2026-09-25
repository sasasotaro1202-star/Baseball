from mlb_game_type import (
    CATEGORIES,
    DEFAULT_TRAINING_CATEGORIES,
    DEFAULT_EVALUATION_CATEGORIES,
    classify_mlb_game,
    parse_categories,
)
from international_game_type import (
    classify_international_game,
    DEFAULT_TRAINING_CATEGORIES as INTL_DEFAULT_TRAINING,
)


def test_mlb_code_taxonomy():
    assert classify_mlb_game("R")["category"] == "regular"
    assert classify_mlb_game("F")["category"] == "wild_card"
    assert classify_mlb_game("D")["category"] == "division_series"
    assert classify_mlb_game("L")["category"] == "league_championship"
    assert classify_mlb_game("W")["category"] == "world_series"
    assert classify_mlb_game("A")["category"] == "allstar"
    assert classify_mlb_game("S")["category"] == "spring_training"
    assert classify_mlb_game("E")["category"] == "exhibition_special"


def test_mlb_description_fallback():
    assert classify_mlb_game("", "Wild Card Series")["category"] == "wild_card"
    assert classify_mlb_game("", "ALDS")["category"] == "division_series"
    assert classify_mlb_game("", "World Series")["category"] == "world_series"


def test_mlb_training_is_conservative():
    assert DEFAULT_TRAINING_CATEGORIES == ("regular",)
    assert "wild_card" in DEFAULT_EVALUATION_CATEGORIES
    assert "allstar" in DEFAULT_EVALUATION_CATEGORIES
    assert "exhibition_special" in DEFAULT_EVALUATION_CATEGORIES
    assert set(DEFAULT_TRAINING_CATEGORIES).issubset(CATEGORIES)


def test_mlb_parse_categories():
    assert parse_categories("regular, wild_card,regular", ()) == ("regular", "wild_card")


def test_international_asian_games_is_explicit():
    out = classify_international_game("Asian Games")
    assert out["category"] == "asian_games"
    assert out["training_default"] is False
    assert "asian_games" in classify_international_game("アジア競技大会")["category"]
    assert INTL_DEFAULT_TRAINING == ()


def test_international_wbsc_events():
    assert classify_international_game("World Baseball Classic")["category"] == "world_baseball_classic"
    assert classify_international_game("Premier12")["category"] == "premier12"
    assert classify_international_game("Olympic Games")["category"] == "olympic_games"
