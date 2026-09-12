"""Canonical score/Low-High contract used by production prediction adapters.

The legacy backtest engine can continue to own its numerical score model. This
module only validates and normalizes its output before it reaches the production
runner. In particular, the four displayed score candidates are allowed to be a
top-k subset, so their probabilities do not have to sum to one; if an "その他"
tail bucket is present, it is explicitly defined as the complete probability
mass outside the enumerated ordinary score cells.
"""
from __future__ import annotations

import math
from typing import Any, Iterable, Mapping


def _finite_probability(value: Any) -> float:
    p = float(value)
    if not math.isfinite(p) or not 0.0 <= p <= 1.0:
        raise ValueError("score probability must be finite and in [0,1]")
    return p


def normalize_score_candidates(candidates: Iterable[Mapping[str, Any]], *, n: int = 4) -> list[dict[str, Any]]:
    """Validate deterministic top-k score candidates without inventing values."""
    if n < 1:
        raise ValueError("n must be positive")
    rows = []
    seen = set()
    for item in candidates:
        if not isinstance(item, Mapping) or "score" not in item or "probability" not in item:
            raise ValueError("each score candidate requires score and probability")
        score = str(item["score"]).strip()
        if not score or score in seen:
            raise ValueError("score candidates must have unique non-empty score labels")
        seen.add(score)
        rows.append({"score": score, "probability": _finite_probability(item["probability"])})
    if not rows:
        raise ValueError("at least one score candidate is required")
    rows.sort(key=lambda x: (-x["probability"], x["score"]))
    return rows[:n]


def validate_low_high(low: Any, high: Any) -> tuple[float, float]:
    l = _finite_probability(low)
    h = _finite_probability(high)
    if abs(l + h - 1.0) > 1e-8:
        raise ValueError("Low/High probabilities must sum to 1")
    return l, h


def classify_with_verified_line(total_runs: int, line: float) -> str:
    """Classify a final total using a verified integer/half-point market line."""
    line = float(line)
    if not math.isfinite(line) or line < 0 or abs(line * 2 - round(line * 2)) > 1e-9:
        raise ValueError("market line must be finite, non-negative, integer or half-point")
    if line.is_integer() and total_runs == int(line):
        return "PUSH"
    return "LOW" if total_runs < line else "HIGH"
