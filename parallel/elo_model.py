"""Chronological Elo walk-forward baseline for baseball-like tables."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import math
import pandas as pd


@dataclass
class EloResult:
    predictions: pd.DataFrame
    final_ratings: Dict[str, float]


def run_elo_walk_forward(
    df: pd.DataFrame,
    *,
    k: float = 20.0,
    home_advantage: float = 25.0,
    initial_rating: float = 1500.0,
) -> EloResult:
    required = {"date", "home_team", "away_team", "home_score", "away_score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    work = df.copy()
    work["_order"] = range(len(work))
    work = work.sort_values(["date", "_order"], kind="stable").reset_index(drop=True)
    ratings: Dict[str, float] = {}
    rows = []

    def rating(team: str) -> float:
        return float(ratings.setdefault(team, initial_rating))

    for _, r in work.iterrows():
        h, a = str(r["home_team"]), str(r["away_team"])
        rh, ra = rating(h), rating(a)
        expected = 1.0 / (1.0 + 10.0 ** (-((rh + home_advantage) - ra) / 400.0))
        hs, as_ = float(r["home_score"]), float(r["away_score"])
        actual = int(hs > as_)
        prediction = "H" if expected >= 0.5 else "A"
        rows.append({
            "date": r["date"],
            "home_team": h,
            "away_team": a,
            "actual_home_win": actual,
            "home_win_probability": expected,
            "prediction": prediction,
            "correct": int((expected >= 0.5) == bool(actual)),
        })
        margin = max(1.0, abs(hs - as_))
        mov_multiplier = math.log1p(margin) * (2.2 / (0.001 * abs((rh + home_advantage) - ra) + 2.2))
        delta = k * mov_multiplier * (actual - expected)
        ratings[h] = rh + delta
        ratings[a] = ra - delta
    return EloResult(predictions=pd.DataFrame(rows), final_ratings=ratings)
