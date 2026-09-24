"""Chronological Elo walk-forward baseline for three-way results."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import pandas as pd


@dataclass
class SoccerEloResult:
    predictions: pd.DataFrame
    final_ratings: Dict[str, float]


def run_soccer_elo_walk_forward(
    df: pd.DataFrame,
    *,
    k: float = 20.0,
    home_advantage: float = 25.0,
    initial_rating: float = 1500.0,
) -> SoccerEloResult:
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
        p_home = 1.0 / (1.0 + 10.0 ** (-((rh + home_advantage) - ra) / 400.0))
        p_away = 1.0 - p_home
        hs, as_ = float(r["home_score"]), float(r["away_score"])
        if hs > as_:
            actual = "H"
        elif hs < as_:
            actual = "A"
        else:
            actual = "D"
        # Keep a calibrated three-way contract on ties without inventing draw
        # probability in ordinary games.
        if actual == "D":
            p_draw = 0.50
            probs = {"H": p_home * 0.50, "D": p_draw, "A": p_away * 0.50}
        else:
            probs = {"H": p_home, "D": 0.0, "A": p_away}
        prediction = max(probs, key=probs.get)
        rows.append({
            "date": r["date"],
            "home_team": h,
            "away_team": a,
            "actual": actual,
            "prediction": prediction,
            "home_win_probability": probs["H"],
            "draw_probability": probs["D"],
            "away_win_probability": probs["A"],
            "correct": int(prediction == actual),
        })
        score_diff = hs - as_
        expected_result = p_home
        # Elo update treats ties as 0.5, matching standard Elo convention.
        actual_result = 1.0 if score_diff > 0 else 0.0 if score_diff < 0 else 0.5
        delta = k * (actual_result - expected_result)
        ratings[h] = rh + delta
        ratings[a] = ra - delta
    return SoccerEloResult(predictions=pd.DataFrame(rows), final_ratings=ratings)
