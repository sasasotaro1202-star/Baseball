#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""International baseball competition scope.

This registry makes international competitions explicit without pretending
they are already available in the MLB/NPB historical datasets.  By default,
international classes are evaluation-only research targets and cannot alter
production model state until their own PIT-safe chronological OOS evidence
exists.
"""
from __future__ import annotations

import os
from typing import Any

CATEGORIES = (
    "world_baseball_classic",
    "premier12",
    "olympic_games",
    "asian_games",
    "pan_american_games",
    "asian_championship",
    "continental_championship",
    "world_cup",
    "other_major_multisport",
    "friendly_exhibition",
    "other_international",
    "unknown",
    "excluded",
)

CATEGORY_TERMS = {
    "world_baseball_classic": ("world baseball classic", "wbc"),
    "premier12": ("premier12", "premier 12"),
    "olympic_games": ("olympic games", "olympics"),
    "asian_games": ("asian games", "アジア競技大会", "アジア大会"),
    "pan_american_games": ("pan american games", "pan-american games"),
    "asian_championship": ("asia baseball championship", "asian baseball championship", "asian championships"),
    "continental_championship": ("continental championship", "baseball asia cup", "baseball asia"),
    "world_cup": ("baseball world cup", "world cup"),
    "other_major_multisport": ("multisport games", "regional multi-sport"),
    "friendly_exhibition": ("friendly", "friendship", "exhibition", "special game"),
}

DEFAULT_TRAINING_CATEGORIES: tuple[str, ...] = ()
DEFAULT_EVALUATION_CATEGORIES = tuple(
    x for x in CATEGORIES if x not in {"unknown", "excluded"}
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
        os.getenv("INTERNATIONAL_TRAINING_GAME_CATEGORIES"),
        DEFAULT_TRAINING_CATEGORIES,
    )


def evaluation_categories() -> tuple[str, ...]:
    return parse_categories(
        os.getenv("INTERNATIONAL_EVALUATION_GAME_CATEGORIES"),
        DEFAULT_EVALUATION_CATEGORIES,
    )


def classify_international_game(competition: Any, description: Any = "") -> dict[str, Any]:
    raw = " ".join(x for x in (competition, description) if x)
    norm = _norm(raw)

    for category, terms in CATEGORY_TERMS.items():
        if any(term in norm for term in terms):
            return {
                "category": category,
                "training_default": category in training_categories(),
                "evaluation_default": category in evaluation_categories(),
                "confidence": "high",
                "raw": raw,
            }

    if not raw.strip():
        category = "unknown"
        confidence = "low"
    else:
        category = "unknown"
        confidence = "low"

    return {
        "category": category,
        "training_default": False,
        "evaluation_default": False,
        "confidence": confidence,
        "raw": raw,
    }


def category_is_training(category: Any) -> bool:
    return str(category or "").strip().lower() in set(training_categories())


def category_is_evaluation(category: Any) -> bool:
    return str(category or "").strip().lower() in set(evaluation_categories())
