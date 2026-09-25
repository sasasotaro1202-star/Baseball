import pandas as pd

from baseball_backtest import BaseballBacktest
from mlb_game_type import (
    CATEGORIES,
    DEFAULT_TRAINING_CATEGORIES,
    DEFAULT_EVALUATION_CATEGORIES,
    classify_mlb_game,
    evaluation_categories,
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
    assert classify_mlb_game("C")["category"] == "championship"
    assert classify_mlb_game("P")["category"] == "postseason"
    assert "championship" in evaluation_categories()
    assert classify_mlb_game("A")["category"] == "allstar"
    assert classify_mlb_game("S")["category"] == "spring_training"
    assert classify_mlb_game("E")["category"] == "exhibition_special"
    assert classify_mlb_game("I")["category"] == "intrasquad"


def test_mlb_description_fallback():
    assert classify_mlb_game("", "Wild Card Series")["category"] == "wild_card"
    assert classify_mlb_game("", "ALDS")["category"] == "division_series"
    assert classify_mlb_game("", "World Series")["category"] == "world_series"


def test_mlb_training_is_conservative():
    assert DEFAULT_TRAINING_CATEGORIES == ("regular",)
    assert "wild_card" in DEFAULT_EVALUATION_CATEGORIES
    assert "allstar" in DEFAULT_EVALUATION_CATEGORIES
    assert "exhibition_special" in DEFAULT_EVALUATION_CATEGORIES
    assert "postseason" in DEFAULT_EVALUATION_CATEGORIES
    assert "intrasquad" not in DEFAULT_EVALUATION_CATEGORIES
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


def test_mlb_postgame_starter_enrichment_is_pit_closed(tmp_path):
    bt = BaseballBacktest(tmp_path)
    games = pd.DataFrame(
        [{
            "game_id": "1",
            "home_starter": "Probable Home",
            "away_starter": "Probable Away",
            "prediction_time_utc": "2026-09-25T16:00:00Z",
            "starter_available_at": "2026-09-25T16:05:00Z",
        }]
    )
    out = bt.enrich_mlb_starters(games)
    assert bool(out.loc[0, "confirmed_starters"]) is False
    assert out.loc[0, "starter_confirmation_state"] == "UNKNOWN"
    assert out.loc[0, "starter_confirmation_source"] == "none"
    assert out.loc[0, "home_starter"] == "Probable Home"
    assert out.loc[0, "away_starter"] == "Probable Away"


def test_mlb_pregame_starter_provenance_can_be_confirmed(tmp_path):
    bt = BaseballBacktest(tmp_path)
    games = pd.DataFrame(
        [{
            "home_pregame_starter": "Confirmed Home",
            "away_pregame_starter": "Confirmed Away",
            "prediction_time_utc": "2026-09-25T16:00:00Z",
            "starter_available_at": "2026-09-25T15:30:00Z",
        }]
    )
    out = bt.enrich_mlb_starters(games)
    assert bool(out.loc[0, "confirmed_starters"]) is True
    assert out.loc[0, "starter_confirmation_state"] == "CONFIRMED"
    assert out.loc[0, "starter_confirmation_source"] == "pregame_snapshot"


def test_mlb_future_prediction_keeps_probable_starters_on_hold(tmp_path):
    bt = BaseballBacktest(tmp_path)
    schedule = pd.DataFrame([{
        "game_id": "g1",
        "home": "Home",
        "away": "Away",
        "home_starter": "Probable Home",
        "away_starter": "Probable Away",
        "confirmed_starters": False,
        "starter_confirmation_state": "PROBABLE",
    }])
    out = bt.build_future_mlb_predictions(schedule)
    assert out.loc[0, "status"] == "保留"
    assert out.loc[0, "reason"] == "両先発の公式確認が揃っていない"
