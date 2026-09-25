import os

from npb_game_type import (
    classify_npb_game,
    parse_categories,
    DEFAULT_TRAINING_CATEGORIES,
    DEFAULT_EVALUATION_CATEGORIES,
)


def test_game_type_taxonomy():
    assert classify_npb_game("セ・リーグ公式戦")["category"] == "regular"
    assert classify_npb_game("パ・リーグ公式戦")["category"] == "regular"
    assert classify_npb_game("セ・パ交流戦")["category"] == "interleague"
    assert classify_npb_game("クライマックスシリーズ ファイナルステージ")["category"] == "climax"
    assert classify_npb_game("日本シリーズ")["category"] == "japan_series"
    assert classify_npb_game("マイナビオールスターゲーム")["category"] == "allstar"
    assert classify_npb_game("親善試合")["category"] == "special"
    assert classify_npb_game("特別試合")["category"] == "special"
    assert classify_npb_game("オープン戦")["category"] == "excluded"
    assert classify_npb_game("二軍公式戦")["category"] == "excluded"
    assert classify_npb_game("")["category"] == "unknown"


def test_default_policy_is_conservative():
    assert DEFAULT_TRAINING_CATEGORIES == ("regular", "interleague")
    assert "climax" in DEFAULT_EVALUATION_CATEGORIES
    assert "japan_series" in DEFAULT_EVALUATION_CATEGORIES
    assert "allstar" in DEFAULT_EVALUATION_CATEGORIES
    assert "special" in DEFAULT_EVALUATION_CATEGORIES


def test_parse_categories_deterministic():
    assert parse_categories("regular, climax,regular", ()) == ("regular", "climax")
    assert parse_categories("", ("regular",)) == ("regular",)
