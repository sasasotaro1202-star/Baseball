"""Strict prediction/result reconciliation audit."""
from __future__ import annotations
import math
import pandas as pd


def _validate_probabilities(predictions: pd.DataFrame, league: str) -> None:
    cols = ["home_win_probability", "away_win_probability"]
    if str(league).upper() == "NPB":
        cols = ["home_win_probability", "draw_probability", "away_win_probability"]
    missing = [c for c in cols if c not in predictions.columns]
    if missing:
        raise ValueError(f"missing probability columns: {missing}")
    probs = predictions[cols].apply(pd.to_numeric, errors="coerce")
    if probs.isna().any().any():
        raise ValueError("non-numeric or missing probability")
    for col in cols:
        if ((probs[col] < 0) | (probs[col] > 1)).any():
            raise ValueError(f"probability outside [0,1]: {col}")
    sums = probs.sum(axis=1)
    if not (sums.sub(1.0).abs() <= 1e-8).all():
        raise ValueError("probability rows must sum to one")


def audit_predictions(predictions: pd.DataFrame, results: pd.DataFrame, *, league: str) -> dict:
    if "event_id" not in predictions.columns or "event_id" not in results.columns:
        raise ValueError("event_id is required in both predictions and results")
    p = predictions.copy()
    r = results.copy()
    if p["event_id"].duplicated().any():
        raise ValueError("duplicate prediction event_id")
    if r["event_id"].duplicated().any():
        raise ValueError("duplicate result event_id")
    _validate_probabilities(p, league)

    pids = set(p["event_id"].astype(str))
    rids = set(r["event_id"].astype(str))
    matched = pids & rids
    return {
        "league": str(league),
        "prediction_rows": int(len(p)),
        "result_rows": int(len(r)),
        "matched_rows": int(len(matched)),
        "unmatched_predictions": int(len(pids - rids)),
        "unmatched_results": int(len(rids - pids)),
        "scored_rows": int(len(matched)),
    }
