#!/usr/bin/env python3
"""Deterministic frozen-holdout integrity/evaluation gate for Baseball OOS results.

The holdout is the latest chronological portion of an already-produced OOS
prediction table. It is never written back into tuning state and is always
reported separately from the earlier development/OOS portion.
"""
from __future__ import annotations
import json, math, sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "frozen_holdout_gate.json"
HOLDOUT_FRAC = 0.15
MIN_HOLDOUT_ROWS = 50

def logloss(actual, p):
    return -math.log(max(min(float(p), 1.0-1e-12), 1e-12))

def ece(conf, correct, bins=10):
    conf = pd.Series(conf, dtype=float)
    correct = pd.Series(correct, dtype=float)
    if len(conf) == 0:
        return float("nan")
    edges = [i / bins for i in range(bins + 1)]
    total = float(len(conf))
    value = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf >= lo) & ((conf < hi) if hi < 1.0 else (conf <= hi))
        if not bool(mask.any()):
            continue
        value += float(mask.mean()) * abs(float(conf[mask].mean()) - float(correct[mask].mean()))
    return value

def evaluate(df):
    if df.empty:
        return {}
    prob_cols = [c for c in ("pred_home", "pred_draw", "pred_away") if c in df.columns]
    if len(prob_cols) == 2:
        probs = df[prob_cols].astype(float)
        pred = probs.iloc[:, 0].to_numpy() >= 0.5
        actual = pd.to_numeric(df["actual"], errors="coerce").to_numpy()
        ll = [logloss(int(a), p if int(a) == 1 else 1.0-p) for a,p in zip(actual, probs.iloc[:,0])]
    elif len(prob_cols) == 3:
        probs = df[prob_cols].astype(float).to_numpy()
        actual = pd.to_numeric(df["actual"], errors="coerce").astype(int).to_numpy()
        chosen = probs.max(axis=1)
        pred = probs.argmax(axis=1)
        ll = [logloss(int(a), probs[i, int(a)]) for i,a in enumerate(actual)]
    else:
        raise ValueError("required probability columns are missing")
    correct = (pred.astype(int) == actual.astype(int)).astype(float)
    return {
        "rows": int(len(df)),
        "accuracy": float(correct.mean()),
        "logloss": float(sum(ll) / len(ll)),
        "ece": float(ece(chosen, correct)),
    }

def main():
    RESULTS.mkdir(exist_ok=True)
    payload = {"status": "DEFERRED", "holdout_fraction": HOLDOUT_FRAC, "min_holdout_rows": MIN_HOLDOUT_ROWS}
    inputs = {"NPB": RESULTS / "npb_backtest_results.csv", "MLB": RESULTS / "mlb_backtest_results.csv"}
    found = False
    for league, path in inputs.items():
        if not path.exists() or path.stat().st_size == 0:
            payload.setdefault("leagues", {})[league] = {"status": "DEFERRED", "reason": "OOS result file missing"}
            continue
        df = pd.read_csv(path)
        if "datetime" not in df.columns:
            raise SystemExit(f"{league}: datetime column missing")
        df = df.sort_values(["datetime", "game_id"] if "game_id" in df.columns else ["datetime"]).reset_index(drop=True)
        n = max(MIN_HOLDOUT_ROWS, int(math.ceil(len(df) * HOLDOUT_FRAC)))
        if len(df) < n + 50:
            payload.setdefault("leagues", {})[league] = {"status": "DEFERRED", "reason": f"too few rows for frozen holdout: {len(df)}"}
            continue
        cut = len(df) - n
        development = df.iloc[:cut].copy()
        holdout = df.iloc[cut:].copy()
        payload.setdefault("leagues", {})[league] = {
            "status": "PASS",
            "development_rows": int(len(development)),
            "holdout_rows": int(len(holdout)),
            "holdout_start": str(holdout["datetime"].iloc[0]),
            "holdout_end": str(holdout["datetime"].iloc[-1]),
            "holdout_metrics": evaluate(holdout),
            "development_metrics": evaluate(development),
            "tuning_rule": "holdout rows are excluded from any candidate selection or promotion tuning",
        }
        found = True
    if found:
        payload["status"] = "PASS"
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))
    if payload["status"] == "DEFERRED":
        raise SystemExit(0)

if __name__ == "__main__":
    main()
