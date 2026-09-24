"""Small, deterministic evaluation metrics."""
from __future__ import annotations
import pandas as pd


def accuracy(correct: pd.Series | list | tuple) -> float:
    s = pd.Series(correct, dtype=float)
    if s.empty:
        return float("nan")
    return float(s.mean())


def base_rate_home_win(predictions: pd.DataFrame, actual_col: str = "actual_home_win") -> float:
    if actual_col not in predictions.columns:
        raise KeyError(f"missing column: {actual_col}")
    s = pd.to_numeric(predictions[actual_col], errors="coerce").dropna()
    if s.empty:
        return float("nan")
    return float(s.mean())
