"""Run the four-stage validation pipeline on the real Baseball backtest data.

This module deliberately uses the existing BaseballBacktest loaders and
walk-forward engine. It does not fabricate observations or declare promotion
without an actual independent holdout.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

from baseball_backtest import BaseballBacktest
from evaluation.target_metrics import hilo_metrics, score_distribution_metrics, score_top4_metrics
from research.adoption_gate import GatePolicy
from research.validation_pipeline import run_validation_pipeline


@dataclass(frozen=True)
class RealDataRun:
    league: str
    source_rows: int
    oos_rows: int
    development_rows: int
    holdout_rows: int
    decision: str
    reasons: tuple[str, ...]


def _binary_from_rows(df: pd.DataFrame, league: str) -> tuple[np.ndarray, np.ndarray]:
    if league == "NPB":
        # Existing walk-forward uses 0=home win, 1=draw, 2=away win.
        y = (df["actual"].astype(int).to_numpy() == 2).astype(int)
        p = df["pred_away"].astype(float).to_numpy()
        return y, p
    y = (df["actual"].astype(int).to_numpy() == 1).astype(int)
    p = df["pred_away"].astype(float).to_numpy()
    return y, p


def _win_metrics(df: pd.DataFrame, league: str) -> dict[str, float]:
    y, p = _binary_from_rows(df, league)
    eps = 1e-15
    pred = (p >= 0.5).astype(int)
    return {
        "Accuracy": float(np.mean(pred == y)),
        "LogLoss": float(-np.mean(y*np.log(np.clip(p, eps, 1-eps)) + (1-y)*np.log(np.clip(1-p, eps, 1-eps)))),
        "Brier": float(np.mean((p-y)**2)),
        "rows": int(len(df)),
    }


def _score_metrics(df: pd.DataFrame) -> dict[str, float]:
    required = ["actual_home_score", "actual_away_score", "pred_home_score", "pred_away_score"]
    if not all(c in df.columns for c in required):
        return {"ScoreMAE": float("nan"), "rows": int(len(df))}
    return score_distribution_metrics(df["actual_home_score"], df["actual_away_score"], df["pred_home_score"], df["pred_away_score"])


def _hilo_metrics(df: pd.DataFrame) -> dict[str, float]:
    if not {"actual_total", "pred_high_prob"}.issubset(df.columns):
        return {"Accuracy": float("nan"), "LogLoss": float("nan"), "Brier": float("nan"), "rows": int(len(df))}
    return hilo_metrics(df["actual_total"].astype(int), df["pred_high_prob"])


def _split(df: pd.DataFrame, development_fraction: float = 0.70) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return df.copy(), df.copy()
    n = max(1, min(len(df)-1, int(len(df) * development_fraction)))
    return df.iloc[:n].copy(), df.iloc[n:].copy()


def run_real_data_validation(
    *,
    data_dir: str | Path = "data",
    league: str,
    mlb_start: int = 2020,
    mlb_end: int = 2026,
    policy: GatePolicy = GatePolicy(),
) -> RealDataRun:
    """Execute Development OOS -> Lock -> Independent Holdout on real data.

    The current production engine supplies the baseline OOS. A candidate is
    represented by the locked model result produced by the same real-data
    walk-forward path until candidate-specific feature/model factories are
    wired in. We therefore refuse promotion when no independent candidate
    result exists rather than treating a duplicate baseline as an improvement.
    """
    bt = BaseballBacktest(Path(data_dir))
    if league == "NPB":
        source = bt.aggregate_npb_games(bt.load_npb_pbp())
    elif league == "MLB":
        source = bt.load_mlb(mlb_start, mlb_end)
    else:
        raise ValueError("league must be NPB or MLB")

    oos = bt.run_walkforward(source, league)
    if oos.empty:
        return RealDataRun(league, len(source), 0, 0, 0, "REJECT", ("no_oos_rows",))

    dev, holdout = _split(oos)
    if holdout.empty:
        return RealDataRun(league, len(source), len(oos), len(dev), 0, "REJECT", ("no_independent_holdout",))

    # Until a distinct candidate factory is supplied, do not silently compare
    # the same model against itself. This function is the real-data wiring
    # point; the explicit rejection is intentional and auditable.
    return RealDataRun(
        league=league,
        source_rows=int(len(source)),
        oos_rows=int(len(oos)),
        development_rows=int(len(dev)),
        holdout_rows=int(len(holdout)),
        decision="REJECT",
        reasons=("candidate_factory_not_connected", "refusing_self_comparison"),
    )
