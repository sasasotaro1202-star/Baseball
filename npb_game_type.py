#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""NPB competition-type taxonomy and conservative training policy.

All detected game types are retained for audit/research.  Only clearly
non-competitive exhibition/farm/education games are hard-excluded.
Ambiguous games are retained but fail closed for model-training eligibility.
"""
from __future__ import annotations
import os
from typing import Any, Dict

CATEGORIES = (
    "regular",
    "interleague",
    "climax",
    "japan_series",
    "allstar",
    "special",
    "unknown",
    "excluded",
)

# Ordered from most-specific to broad fallback.
CATEGORY_TERMS = {
    "climax": ("クライマックスシリーズ", "クライマックス", "CSファースト", "CSファイナル", "climax series", "climax"),
    "japan_series": ("日本シリーズ", "日本選手権シリーズ", "japan series"),
    "allstar": ("オールスター", "all-star", "all star", "フレッシュオールスター", "fresh all-star"),
    "special": (
        "親善試合", "親善試合", "特別試合", "壮行試合", "チャリティー試合",
        "記念試合", "エキシビション", "special game", "friendly", "friendship",
    ),
    "interleague": ("交流戦", "セ・パ交流戦", "interleague"),
    "regular": ("セ・リーグ公式戦", "パ・リーグ公式戦", "公式戦", "レギュラーシーズン", "regular season"),
}
EXCLUDED_TERMS = (
    "オープン戦", "オープン戦", "春季キャンプ", "キャンプ", "ファーム",
    "二軍", "二軍公式戦", "教育リーグ", "春季教育リーグ", "秋季教育リーグ",
    "練習試合", "紅白戦", "practice", "training", "minor league", "farm",
    "open game", "exhibition",
)

# Production-safe baseline: regular season + interleague only until each
# postseason/special class independently demonstrates OOS benefit.
DEFAULT_TRAINING_CATEGORIES = ("regular", "interleague")
DEFAULT_EVALUATION_CATEGORIES = ("regular", "interleague", "climax", "japan_series", "allstar", "special")


def _norm(v: Any) -> str:
    return " ".join(str(v or "").strip().lower().split())


def classify_npb_game(game_type: Any, *, raw_context: Any = "") -> Dict[str, Any]:
    raw = str(game_type or "").strip()
    text = _norm(" ".join(x for x in (raw, raw_context) if x))
    if any(term.lower() in text for term in EXCLUDED_TERMS):
        return {"category": "excluded", "training_default": False, "evaluation_default": False, "confidence": "high", "raw": raw}
    for category, terms in CATEGORY_TERMS.items():
        if any(term.lower() in text for term in terms):
            train = category in DEFAULT_TRAINING_CATEGORIES
            evaluate = category in DEFAULT_EVALUATION_CATEGORIES
            return {"category": category, "training_default": train, "evaluation_default": evaluate, "confidence": "high", "raw": raw}
    if not raw:
        return {"category": "unknown", "training_default": False, "evaluation_default": False, "confidence": "low", "raw": raw}
    return {"category": "unknown", "training_default": False, "evaluation_default": False, "confidence": "low", "raw": raw}


def parse_categories(value: str | None, default: tuple[str, ...]) -> tuple[str, ...]:
    if value is None or not str(value).strip():
        return tuple(default)
    out=[]
    for item in str(value).split(","):
        x=item.strip().lower()
        if x and x not in out:
            out.append(x)
    return tuple(out)


def training_categories() -> tuple[str, ...]:
    return parse_categories(os.getenv("NPB_TRAINING_GAME_CATEGORIES"), DEFAULT_TRAINING_CATEGORIES)


def evaluation_categories() -> tuple[str, ...]:
    return parse_categories(os.getenv("NPB_EVALUATION_GAME_CATEGORIES"), DEFAULT_EVALUATION_CATEGORIES)


def category_is_training(category: Any) -> bool:
    return str(category or "").strip().lower() in set(training_categories())


def category_is_evaluation(category: Any) -> bool:
    return str(category or "").strip().lower() in set(evaluation_categories())


def is_excluded_game_type(game_type: Any, *, raw_context: Any = "") -> bool:
    return classify_npb_game(game_type, raw_context=raw_context)["category"] == "excluded"
