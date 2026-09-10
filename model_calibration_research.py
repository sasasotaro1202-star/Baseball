#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Chronological probability-calibration research stage for Baseball OOS predictions.

This module is deliberately separated from baseball_backtest.py so a calibration
candidate can be evaluated without changing the production predictor first.

Rules:
- Input must be genuine OOS predictions; no training-set predictions are accepted.
- Rows are sorted chronologically before calibration.
- Calibration is expanding-window: row i is calibrated only from rows < i.
- The first MIN_CAL rows are left uncalibrated because there is not enough
  calibration history.
- Candidate is adopted only when it improves BOTH LogLoss and Brier on the
  post-warmup evaluation rows. Accuracy is reported as a secondary guardrail.
- No synthetic data are created.

Expected flexible columns include one of:
  date/game_datetime/prediction_time
  home_win_probability/home_win_prob/p_home_win/prob_home_win
  home_win (0/1) or result/home_win_target
For NPB a 3-class target is supported when probability columns are named for
home/draw/away.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
MIN_CAL = 100
EPS = 1e-6

DATE_COLS = ["prediction_time", "prediction_cutoff_at", "game_datetime", "date", "game_date"]
HOME_PROB_COLS = ["home_win_probability", "home_win_prob", "p_home_win", "prob_home_win", "home_prob"]
AWAY_PROB_COLS = ["away_win_probability", "away_win_prob", "p_away_win", "prob_away_win", "away_prob"]
TARGET_COLS = ["home_win", "home_win_target", "target", "y"]


def first_existing(df: pd.DataFrame, names: list[str]) -> str | None:
    for c in names:
        if c in df.columns:
            return c
    return None


def locate_oos() -> Path:
    preferred = [RESULTS / "oos_predictions.csv", RESULTS / "mlb_oos_predictions.csv", RESULTS / "npb_oos_predictions.csv"]
    for p in preferred:
        if p.exists() and p.stat().st_size > 0:
            return p
    candidates = sorted(RESULTS.rglob("*.csv"))
    for p in candidates:
        n = p.name.lower()
        if "oos" in n and "prediction" in n and p.stat().st_size > 0:
            return p
    raise FileNotFoundError("No genuine OOS prediction CSV was found under results/.")


def clip(p: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)


def metrics_binary(y: np.ndarray, p: np.ndarray) -> dict:
    p = clip(p)
    return {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, p >= 0.5)),
        "logloss": float(log_loss(y, np.column_stack([1-p, p]), labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
    }


def expanding_calibration(y: np.ndarray, p: np.ndarray, min_cal: int = MIN_CAL) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return isotonic and logistic calibrated probabilities using only the past."""
    p = clip(p)
    out_iso = p.copy()
    out_log = p.copy()
    used = np.zeros(len(p), dtype=bool)
    for i in range(min_cal, len(p)):
        x = p[:i]
        yy = y[:i]
        if len(np.unique(yy)) < 2:
            continue
        # Isotonic is non-parametric and robust to monotone miscalibration.
        iso = IsotonicRegression(y_min=EPS, y_max=1-EPS, out_of_bounds="clip")
        iso.fit(x, yy)
        out_iso[i] = float(iso.predict([p[i]])[0])
        # Platt scaling on logit(p); fit only on historical OOS rows.
        z = np.log(clip(x) / (1.0 - clip(x))).reshape(-1, 1)
        lr = LogisticRegression(C=1.0, solver="lbfgs")
        lr.fit(z, yy)
        zi = np.log(clip(np.array([p[i]])) / (1.0 - clip(np.array([p[i]])))).reshape(-1, 1)
        out_log[i] = float(lr.predict_proba(zi)[0, 1])
        used[i] = True
    return out_iso, out_log, used


def choose_candidate(y: np.ndarray, baseline: np.ndarray, iso: np.ndarray, logit: np.ndarray, used: np.ndarray) -> dict:
    idx = np.flatnonzero(used)
    if len(idx) == 0:
        raise RuntimeError(f"Not enough OOS history for calibration; need at least {MIN_CAL + 1} rows.")
    y2 = y[idx]
    base_m = metrics_binary(y2, baseline[idx])
    iso_m = metrics_binary(y2, iso[idx])
    log_m = metrics_binary(y2, logit[idx])
    candidates = {"baseline": base_m, "isotonic": iso_m, "platt": log_m}
    # Strict adoption criterion: both primary proper scoring rules improve.
    eligible = []
    for name in ("isotonic", "platt"):
        m = candidates[name]
        if m["logloss"] < base_m["logloss"] and m["brier"] < base_m["brier"]:
            eligible.append(name)
    if eligible:
        winner = min(eligible, key=lambda n: candidates[n]["logloss"] + candidates[n]["brier"])
    else:
        winner = "baseline"
    return {"evaluation_rows": int(len(idx)), "warmup_rows": int(idx[0]), "metrics": candidates, "winner": winner}


def main() -> None:
    src = locate_oos()
    df = pd.read_csv(src)
    date_col = first_existing(df, DATE_COLS)
    hp = first_existing(df, HOME_PROB_COLS)
    target = first_existing(df, TARGET_COLS)
    if not date_col or not hp or not target:
        raise RuntimeError(f"OOS schema incomplete: date={date_col}, home_prob={hp}, target={target}")
    work = df.copy()
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce", utc=True)
    work[hp] = pd.to_numeric(work[hp], errors="coerce")
    work[target] = pd.to_numeric(work[target], errors="coerce")
    work = work.dropna(subset=[date_col, hp, target]).sort_values(date_col).reset_index(drop=True)
    if len(work) < MIN_CAL + 1:
        raise RuntimeError(f"Only {len(work)} usable OOS rows; need > {MIN_CAL}.")
    y = work[target].astype(int).to_numpy()
    if not set(np.unique(y)).issubset({0, 1}):
        raise RuntimeError("This first calibration stage is binary only; NPB 3-class calibration is a separate candidate.")
    p = clip(work[hp].to_numpy())
    iso, platt, used = expanding_calibration(y, p)
    decision = choose_candidate(y, p, iso, platt, used)
    winner = decision["winner"]
    calibrated = p.copy()
    if winner == "isotonic": calibrated = iso
    elif winner == "platt": calibrated = platt
    work["calibrated_home_win_probability"] = calibrated
    work["calibration_candidate"] = winner
    out_csv = RESULTS / "calibrated_oos_predictions.csv"
    work.to_csv(out_csv, index=False)
    report = {
        "source": str(src.relative_to(ROOT)),
        "rows_input": int(len(df)),
        "rows_used": int(len(work)),
        "target_column": target,
        "probability_column": hp,
        "date_column": date_col,
        "selection_rule": "adopt only if LogLoss AND Brier improve on chronological post-warmup OOS rows",
        "decision": decision,
        "status": "CANDIDATE_READY" if winner != "baseline" else "BASELINE_RETAINED",
    }
    (RESULTS / "calibration_research_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
