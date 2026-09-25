#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MLB competition-type taxonomy with fail-closed training defaults.

Every recognized game type remains available for audit/OOS research.  Only
Regular Season is enabled for production training by default.  Postseason,
All-Star, Spring Training and Exhibition/Special classes can be evaluated
individually and opted into training only through explicit environment
configuration after chronological OOS validation.
"""
from __future__ import annotations

import os
from typing import Any, Dict

CATEGORIES = (
    "regular",
    "wild_card",
    "division_series",
    "league_championship",
    "world_series",
    "championship",
    "postseason",
    "allstar",
    "spring_training",
    "exhibition_special",
    "intrasquad",
    "unknown",
    "excluded",
)

# Current MLB Stats API gameTypes. Source verified against:
# GET https://statsapi.mlb.com/api/v1/gameTypes
GAME_TYPE_CODE_MAP = {
    "R": "regular",
    "F": "wild_card",
    "D": "division_series",
    "L": "league_championship",
    "W": "world_series",
    "C": "championship",
    "P": "postseason",
    "A": "allstar",
    "S": "spring_training",
    "E": "exhibition_special",
    "I": "intrasquad",
}

CATEGORY_TERMS = {
    "wild_card": ("wild card", "wild-card"),
    "division_series": ("division series", "alds", "nlds"),
    "league_championship": ("league championship", "championship series", "alcs", "nlcs"),
    "world_series": ("world series",),
    "allstar": ("all-star", "all star", "allstar", "midsummer classic"),
    "spring_training": ("spring training", "spring exhibition"),
    "exhibition_special": (
        "exhibition", "friendly", "friendship", "special game",
        "special series", "charity game", "international exhibition",
    ),
    "regular": ("regular season",),
    "championship": ("championship",),
    "postseason": ("postseason", "playoffs"),
}

DEFAULT_TRAINING_CATEGORIES = ("regular",)
DEFAULT_EVALUATION_CATEGORIES = (
    "regular",
    "wild_card",
    "division_series",
    "league_championship",
    "world_series",
    "championship",
    "allstar",
    "spring_training",
    "exhibition_special",
    "postseason",
)


def _norm(v: Any) -> str:
    return " ".join(str(v or "").strip().lower().split())


def parse_categories(value: str | None, default: tuple[str, ...]) -> tuple[str, ...]:
    if value is None or not str(value).strip():
        return tuple(default)
    out = []
    for item in str(value).split(","):
        x = item.strip().lower()
        if x and x not in out:
            out.append(x)
    return tuple(out)


def training_categories() -> tuple[str, ...]:
    return parse_categories(
        os.getenv("MLB_TRAINING_GAME_CATEGORIES"),
        DEFAULT_TRAINING_CATEGORIES,
    )


def evaluation_categories() -> tuple[str, ...]:
    return parse_categories(
        os.getenv("MLB_EVALUATION_GAME_CATEGORIES"),
        DEFAULT_EVALUATION_CATEGORIES,
    )


def classify_mlb_game(
    game_type_code: Any = "",
    series_description: Any = "",
    raw_context: Any = "",
) -> Dict[str, Any]:
    code = str(game_type_code or "").strip().upper()
    series = str(series_description or "").strip()
    raw = " ".join(x for x in (series, raw_context) if x)
    norm = _norm(raw)

    if any(term in norm for term in ("minor league", "minor-league", "farm league")):
        return {
            "category": "excluded",
            "training_default": False,
            "evaluation_default": False,
            "confidence": "high",
            "raw": series or code,
        }

    category = GAME_TYPE_CODE_MAP.get(code)
    if category is None:
        for candidate, terms in CATEGORY_TERMS.items():
            if any(term in norm for term in terms):
                category = candidate
                break

    if category is None:
        category = "unknown" if not code and not series and not raw else "unknown"

    return {
        "category": category,
        "training_default": category in DEFAULT_TRAINING_CATEGORIES,
        "evaluation_default": category in DEFAULT_EVALUATION_CATEGORIES,
        "confidence": "high" if code in GAME_TYPE_CODE_MAP or category != "unknown" else "low",
        "raw": series or code or raw,
    }


def category_is_training(category: Any) -> bool:
    return str(category or "").strip().lower() in set(training_categories())


def category_is_evaluation(category: Any) -> bool:
    return str(category or "").strip().lower() in set(evaluation_categories())
