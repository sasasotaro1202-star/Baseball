#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unified competition taxonomy for the baseball prediction system.

The taxonomy is deliberately broader than the current production training
set. Every recognized competition is retained for audit/OOS analysis, while
training eligibility is controlled independently per league/competition.

Key safety rule: unknown or ambiguous competition types are not training data
by default and remain evaluation-deferred until classified.
"""
from __future__ import annotations

import os
import re
from typing import Any, Dict, Iterable, Mapping

# Canonical competition classes used in artifacts.
CATEGORIES = {
    "NPB": (
        "regular",
        "interleague",
        "climax",
        "japan_series",
        "allstar",
        "special",
        "exhibition",
        "farm",
        "unknown",
    ),
    "MLB": (
        "regular",
        "wild_card",
        "division_series",
        "league_championship_series",
        "world_series",
        "postseason",
        "allstar",
        "spring_training",
        "exhibition",
        "intrasquad",
        "championship",
        "unknown",
    ),
    "INTERNATIONAL": (
        "wbc",
        "premier12",
        "olympics",
        "asian_games",
        "asian_professional_baseball_championship",
        "international_friendly",
        "international_special",
        "other_international",
        "unknown",
    ),
}

# Production-safe defaults:
# - NPB: regular + interleague learn; special/postseason/event classes are OOS
#   evaluation candidates.
# - MLB: regular season learns; postseason/all-star/special classes are OOS
#   evaluation candidates.
# - INTERNATIONAL: do not mix national-team/tournament data into club-league
#   training by default; evaluate separately first.
DEFAULT_TRAINING = {
    "NPB": ("regular", "interleague"),
    "MLB": ("regular",),
    "INTERNATIONAL": (),
}
DEFAULT_EVALUATION = {
    "NPB": ("regular", "interleague", "climax", "japan_series", "allstar", "special"),
    "MLB": (
        "regular",
        "wild_card",
        "division_series",
        "league_championship_series",
        "world_series",
        "postseason",
        "allstar",
        "championship",
        "exhibition",
    ),
    "INTERNATIONAL": tuple(
        x for x in CATEGORIES["INTERNATIONAL"] if x != "unknown"
    ),
}

# NPB labels and common Japanese site text.
_NPB_TERMS = {
    "climax": ("クライマックスシリーズ", "クライマックス", "CSファースト", "CSファイナル", "climax series"),
    "japan_series": ("日本シリーズ", "日本選手権シリーズ", "japan series"),
    "allstar": ("オールスター", "all-star", "all star", "フレッシュオールスター", "fresh all-star"),
    "special": (
        "親善試合", "特別試合", "壮行試合", "チャリティー試合", "記念試合",
        "special game", "friendly", "friendship",
    ),
    "interleague": ("交流戦", "セ・パ交流戦", "interleague"),
    "regular": ("セ・リーグ公式戦", "パ・リーグ公式戦", "公式戦", "レギュラーシーズン", "regular season"),
    "exhibition": ("オープン戦", "exhibition", "open game"),
    "farm": ("ファーム", "二軍", "教育リーグ", "春季教育リーグ", "秋季教育リーグ", "minor league", "farm"),
}

# MLB Stats API / documented game-type codes.
_MLB_CODE = {
    "R": "regular",
    "F": "wild_card",
    "D": "division_series",
    "L": "league_championship_series",
    "W": "world_series",
    "A": "allstar",
    "S": "spring_training",
    "E": "exhibition",
    "I": "intrasquad",
    "C": "championship",
    "P": "postseason",
}

_INTERNATIONAL_TERMS = {
    "wbc": ("world baseball classic", "baseball classic", "ワールド・ベースボール・クラシック", "wbc"),
    "premier12": ("wbsc premier12", "premier12", "premier 12", "プレミア12"),
    "olympics": ("olympic baseball", "olympics", "オリンピック", "五輪"),
    "asian_games": ("asian games", "アジア大会", "asiad", "aichi-nagoya asian games", "愛知・名古屋アジア競技大会"),
    "asian_professional_baseball_championship": (
        "asia professional baseball championship",
        "asia professional baseball championship",
        "asian professional baseball championship",
        "apbc",
        "アジア プロ野球チャンピオンシップ",
    ),
    "international_friendly": ("international friendly", "friendly", "親善試合", "日米野球", "強化試合"),
    "international_special": ("international special", "special game", "特別試合", "壮行試合"),
}


def _norm(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _contains_any(text: str, terms: Iterable[str]) -> bool:
    return any(t in text for t in terms)


def classify_game(
    league: str,
    *,
    game_type_code: Any = "",
    game_type: Any = "",
    series_description: Any = "",
    competition: Any = "",
    raw_context: Any = "",
) -> Dict[str, Any]:
    """Return canonical competition classification and default policy."""
    lg = _norm(league).upper()
    text = _norm(" ".join(
        str(x or "") for x in
        (game_type, series_description, competition, raw_context)
    ))

    if lg == "MLB":
        code = str(game_type_code or "").strip().upper()
        if code in _MLB_CODE:
            category = _MLB_CODE[code]
        else:
            # Fall back conservatively to descriptive fields.
            if "wild card" in text:
                category = "wild_card"
            elif "division series" in text:
                category = "division_series"
            elif "league championship series" in text or "lcs" in text:
                category = "league_championship_series"
            elif "world series" in text:
                category = "world_series"
            elif "postseason" in text:
                category = "postseason"
            elif "all-star" in text or "all star" in text:
                category = "allstar"
            elif "spring training" in text:
                category = "spring_training"
            elif "exhibition" in text:
                category = "exhibition"
            elif "intrasquad" in text:
                category = "intrasquad"
            elif "championship" in text:
                category = "championship"
            elif "regular" in text:
                category = "regular"
            else:
                category = "unknown"
        return _policy_result("MLB", category, code or text)

    if lg == "NPB":
        # Specific classes first to avoid "公式戦" swallowing postseason text.
        for category in (
            "climax", "japan_series", "allstar", "special",
            "interleague", "regular", "exhibition", "farm",
        ):
            if _contains_any(text, tuple(_norm(x) for x in _NPB_TERMS[category])):
                return _policy_result("NPB", category, str(game_type or ""))
        return _policy_result("NPB", "unknown" if not text else "unknown", str(game_type or ""))

    if lg in {"INTERNATIONAL", "INTL", "INTERNATIONAL_BASEBALL"}:
        for category, terms in _INTERNATIONAL_TERMS.items():
            if _contains_any(text, tuple(_norm(x) for x in terms)):
                return _policy_result("INTERNATIONAL", category, str(competition or game_type or ""))
        return _policy_result("INTERNATIONAL", "unknown", str(competition or game_type or ""))

    return _policy_result("INTERNATIONAL", "unknown", str(competition or game_type or ""), recognized_league=False)


def _split_env(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    out = []
    for item in raw.split(","):
        value = item.strip().lower()
        if value and value not in out:
            out.append(value)
    return tuple(out)


def training_categories(league: str) -> tuple[str, ...]:
    lg = _norm(league).upper()
    if lg == "NPB":
        return _split_env("BASEBALL_NPB_TRAINING_GAME_CATEGORIES", DEFAULT_TRAINING["NPB"])
    if lg == "MLB":
        return _split_env("BASEBALL_MLB_TRAINING_GAME_CATEGORIES", DEFAULT_TRAINING["MLB"])
    return _split_env("BASEBALL_INTL_TRAINING_GAME_CATEGORIES", DEFAULT_TRAINING["INTERNATIONAL"])


def evaluation_categories(league: str) -> tuple[str, ...]:
    lg = _norm(league).upper()
    if lg == "NPB":
        return _split_env("BASEBALL_NPB_EVALUATION_GAME_CATEGORIES", DEFAULT_EVALUATION["NPB"])
    if lg == "MLB":
        return _split_env("BASEBALL_MLB_EVALUATION_GAME_CATEGORIES", DEFAULT_EVALUATION["MLB"])
    return _split_env("BASEBALL_INTL_EVALUATION_GAME_CATEGORIES", DEFAULT_EVALUATION["INTERNATIONAL"])


def category_is_training(league: str, category: Any) -> bool:
    return str(category or "").strip().lower() in set(training_categories(league))


def category_is_evaluation(league: str, category: Any) -> bool:
    return str(category or "").strip().lower() in set(evaluation_categories(league))


def _policy_result(
    league: str,
    category: str,
    raw: str,
    *,
    recognized_league: bool = True,
) -> Dict[str, Any]:
    trainable = category in set(training_categories(league))
    evaluable = category in set(evaluation_categories(league))
    return {
        "league": league,
        "category": category,
        "training_default": bool(trainable),
        "evaluation_default": bool(evaluable),
        "recognized": bool(recognized_league and category != "unknown"),
        "raw": raw,
    }


def taxonomy_schema() -> Mapping[str, tuple[str, ...]]:
    return {k: tuple(v) for k, v in CATEGORIES.items()}
