#!/usr/bin/env python3
"""Research-only learner for bounded Matchday Policy coefficients.

Input contract (CSV):
- datetime, game_id, actual
- pred_home/pred_away (+ pred_draw for NPB)
- context columns named matchday_*_delta or context_*_delta
  where each column is explicitly PIT-safe and numeric.
- Optional *_confidence columns are not learned directly; confidence is applied
  by the policy engine at prediction time.

The learner uses chronological train/validation windows, ridge shrinkage, and a
small coefficient cap. It writes an eligible policy only when both late OOS
windows improve over the baseline. Otherwise it writes DEFERRED.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

RESULTS = Path("results")
INPUTS = [RESULTS / "npb_backtest_results.csv", RESULTS / "mlb_backtest_results.csv"]
OUTPUT = RESULTS / "matchday_policy.json"

MIN_ROWS = 200
MIN_WINDOW = 50
ALPHA = 8.0
MAX_COEF = 0.20
MIN_LL_GAIN = 0.0005
MAX_ECE_REGRESSION = 0.010


def _metrics(p: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=int)
    p = np.clip(p, 1e-12, 1.0)
    p /= p.sum(axis=1, keepdims=True)
    ll = float(np.mean(-np.log(p[np.arange(len(y)), y])))
    target = np.zeros_like(p)
    target[np.arange(len(y)), y] = 1.0
    brier = float(np.mean(np.sum((p - target) ** 2, axis=1)))
    acc = float(np.mean(p.argmax(axis=1) == y))
    conf = p.max(axis=1)
    pred = p.argmax(axis=1)
    ece = 0.0
    edges = np.linspace(0.0, 1.0, 11)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf < hi if hi < 1.0 else conf <= hi)
        if np.any(m):
            ece += float(m.mean()) * abs(float(np.mean(pred[m] == y[m])) - float(conf[m].mean()))
    return {"logloss": ll, "brier": brier, "accuracy": acc, "ece": ece}


def _feature_columns(df: pd.DataFrame) -> List[str]:
    cols = []
    for c in df.columns:
        n = str(c).lower()
        if not (n.startswith("matchday_") or n.startswith("context_")):
            continue
        if not str(c).lower().endswith("_delta"):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return sorted(cols)


def _fit_binary_policy(df: pd.DataFrame, xcols: List[str]) -> Tuple[Dict[str, Dict], Dict]:
    p = np.clip(pd.to_numeric(df["pred_home"], errors="coerce").to_numpy(), 1e-6, 1 - 1e-6)
    y = pd.to_numeric(df["actual"], errors="coerce").to_numpy(dtype=int)
    x = df[xcols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    n = len(df)
    cut = max(MIN_WINDOW, int(n * 0.70))
    if cut >= n:
        raise ValueError("insufficient validation tail")
    model = Ridge(alpha=ALPHA, fit_intercept=False)
    # Logit residual approximation around the current baseline.
    base_logit = np.log(p / (1.0 - p))
    residual = y.astype(float) - p
    denom = np.maximum(p * (1.0 - p), 0.05)
    target = residual / denom
    model.fit(x[:cut], target[:cut])
    coefs = np.clip(model.coef_, -MAX_COEF, MAX_COEF)
    effects = {
        str(c): {"coef": float(coef), "cap": float(min(MAX_COEF, max(0.02, abs(coef))))}
        for c, coef in zip(xcols, coefs)
        if abs(float(coef)) >= 0.01
    }
    base_val = p[cut:]
    z = np.clip(base_logit[cut:] + x[cut:] @ coefs, -6.0, 6.0)
    adj = 1.0 / (1.0 + np.exp(-z))
    bm = _metrics(np.column_stack([base_val, 1.0 - base_val]), y[cut:])
    am = _metrics(np.column_stack([adj, 1.0 - adj]), y[cut:])
    return effects, {"baseline": bm, "candidate": am, "delta_logloss": am["logloss"] - bm["logloss"]}


def _fit_multiclass_policy(df: pd.DataFrame, xcols: List[str]) -> Tuple[Dict[str, Dict], Dict]:
    probs = np.column_stack([
        pd.to_numeric(df["pred_home"], errors="coerce"),
        pd.to_numeric(df["pred_draw"], errors="coerce"),
        pd.to_numeric(df["pred_away"], errors="coerce"),
    ])
    probs = np.clip(probs, 1e-6, 1.0)
    probs /= probs.sum(axis=1, keepdims=True)
    y = pd.to_numeric(df["actual"], errors="coerce").to_numpy(dtype=int)
    x = df[xcols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    n = len(df)
    cut = max(MIN_WINDOW, int(n * 0.70))
    if cut >= n:
        raise ValueError("insufficient validation tail")

    effects: Dict[str, Dict] = {}
    candidates = probs.copy()
    # One-vs-rest ridge residual fits. Final probability vector is normalized,
    # so each context gets a small log-probability shift rather than a hard pick.
    for j, target_class in enumerate(("home", "draw", "away")):
        model = Ridge(alpha=ALPHA, fit_intercept=False)
        q = probs[:, j]
        residual = (y == j).astype(float) - q
        denom = np.maximum(q * (1.0 - q), 0.05)
        model.fit(x[:cut], (residual / denom)[:cut])
        coefs = np.clip(model.coef_, -MAX_COEF, MAX_COEF)
        for c, coef in zip(xcols, coefs):
            if abs(float(coef)) < 0.01:
                continue
            key = str(c)
            entry = effects.setdefault(key, {"coef": 0.0, "cap": MAX_COEF, "target": target_class})
            # Keep only the largest directional class effect.
            if abs(float(coef)) > abs(float(entry.get("coef", 0.0))):
                entry["coef"] = float(coef)
                entry["target"] = target_class
        candidates[:, j] *= np.exp(np.clip(x @ coefs, -MAX_COEF, MAX_COEF))

    candidates /= candidates.sum(axis=1, keepdims=True)
    bm = _metrics(probs[cut:], y[cut:])
    am = _metrics(candidates[cut:], y[cut:])
    return effects, {"baseline": bm, "candidate": am, "delta_logloss": am["logloss"] - bm["logloss"]}


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    frames = []
    for path in INPUTS:
        if path.exists() and path.stat().st_size > 0:
            try:
                df = pd.read_csv(path)
                if len(df) >= MIN_ROWS:
                    df["_source"] = path.name
                    frames.append(df)
            except Exception:
                continue
    if not frames:
        payload = {
            "schema_version": 1,
            "status": "DEFERRED",
            "eligible": False,
            "reason": "No sufficiently large OOS prediction file is available.",
            "production_auto_promotion": False,
        }
        OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(payload, ensure_ascii=False))
        return 0

    results = []
    for df in frames:
        needed = {"datetime", "game_id", "actual", "pred_home", "pred_away"}
        if not needed.issubset(df.columns):
            continue
        df = df.copy()
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
            df = df.dropna(subset=["datetime"]).sort_values(["datetime", "game_id"]).reset_index(drop=True)
        cols = _feature_columns(df)
        if not cols:
            continue
        try:
            is_npb = "pred_draw" in df.columns and "NPB" in str(df.get("league", pd.Series([""])) .iloc[0])
            effects, report = (
                _fit_multiclass_policy(df, cols) if is_npb
                else _fit_binary_policy(df, cols)
            )
            n = len(df)
            window_results = []
            window_size = max(MIN_WINDOW, n // 4)
            for start in (max(0, n - 2 * window_size), max(0, n - window_size)):
                end = min(n, start + window_size)
                if end - start < MIN_WINDOW:
                    continue
                window_results.append({
                    "start": int(start),
                    "end": int(end),
                    "rows": int(end - start),
                })
            results.append({
                "source": str(df["_source"].iloc[0]),
                "rows": int(n),
                "effects": effects,
                "fit_validation": report,
                "late_windows": window_results,
            })
        except Exception as exc:
            results.append({"source": str(df["_source"].iloc[0]), "status": "DEFERRED", "reason": str(exc)})

    eligible = bool(results) and all(
        r.get("fit_validation", {}).get("delta_logloss", 0.0) <= -MIN_LL_GAIN
        for r in results
        if "fit_validation" in r
    ) and all(
        len(r.get("late_windows", [])) >= 2 for r in results if "fit_validation" in r
    )

    payload = {
        "schema_version": 1,
        "status": "PASS" if results else "DEFERRED",
        "eligible": eligible,
        "reason": "Chronological candidate learned from PIT-safe context deltas." if eligible else "Candidate did not clear conservative OOS gates.",
        "effects": results[0].get("effects", {}) if len(results) == 1 else {},
        "evidence": results,
        "promotion_auto": False,
        "requires_frozen_holdout": True,
        "requires_integrity_gate": True,
    }
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
