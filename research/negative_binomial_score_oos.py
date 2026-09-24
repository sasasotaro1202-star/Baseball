#!/usr/bin/env python3
"""Chronological OOS Negative-Binomial score-distribution challenger.

Poisson is the incumbent run-distribution assumption. This challenger estimates
over-dispersion on an earlier chronological segment, freezes alpha, then evaluates
the untouched later segment. It never changes production state.
"""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
import pandas as pd

RESULTS = Path("results")
MIN_ROWS = 160
FIT_FRAC = 0.60
MIN_IMPROVEMENT = 0.001
ALPHAS = np.concatenate(([0.0], np.geomspace(1e-4, 5.0, 80)))


def poisson_nll(y, mu):
    mu = np.clip(np.asarray(mu, float), 1e-6, 50.0)
    yy = np.asarray(y, float)
    return float(np.mean(mu - yy * np.log(mu) + np.array([math.lgamma(v + 1.0) for v in yy])))


def nb_nll(y, mu, alpha):
    if alpha <= 0.0:
        return poisson_nll(y, mu)
    mu = np.clip(np.asarray(mu, float), 1e-6, 50.0)
    yy = np.asarray(y, float)
    r = 1.0 / float(alpha)
    p = r / (r + mu)
    logpmf = (
        np.array([math.lgamma(v + r) - math.lgamma(r) - math.lgamma(v + 1.0) for v in yy])
        + r * np.log(np.clip(p, 1e-12, 1.0))
        + yy * np.log(np.clip(1.0 - p, 1e-12, 1.0))
    )
    return float(-np.mean(logpmf))


def fit_alpha(y_home, mu_home, y_away, mu_away):
    best_alpha, best_loss = 0.0, float("inf")
    for alpha in ALPHAS:
        loss = 0.5 * (nb_nll(y_home, mu_home, float(alpha)) + nb_nll(y_away, mu_away, float(alpha)))
        if loss < best_loss:
            best_loss, best_alpha = loss, float(alpha)
    return best_alpha


def evaluate(path):
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        return {"status": "DEFERRED", "reason": f"unreadable: {exc}", "rows": 0}
    required = {"datetime", "home_score", "away_score", "lambda_home", "lambda_away"}
    if not required.issubset(df.columns):
        return {"status": "DEFERRED", "reason": "required score columns missing", "rows": int(len(df))}
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
    for c in required - {"datetime"}:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    sort_cols = ["datetime"] + (["game_id"] if "game_id" in df.columns else [])
    df = df.dropna(subset=list(required)).sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    df = df[(df.home_score >= 0) & (df.away_score >= 0) & (df.lambda_home > 0) & (df.lambda_away > 0)]
    if len(df) < MIN_ROWS:
        return {"status": "DEFERRED", "reason": f"insufficient rows: {len(df)} < {MIN_ROWS}", "rows": int(len(df))}
    cut = min(max(int(len(df) * FIT_FRAC), 80), len(df) - 50)
    fit, test = df.iloc[:cut], df.iloc[cut:]
    alpha = fit_alpha(
        fit.home_score.to_numpy(float), fit.lambda_home.to_numpy(float),
        fit.away_score.to_numpy(float), fit.lambda_away.to_numpy(float),
    )
    yh, ya = test.home_score.to_numpy(float), test.away_score.to_numpy(float)
    mh, ma = test.lambda_home.to_numpy(float), test.lambda_away.to_numpy(float)
    poisson_ll = 0.5 * (poisson_nll(yh, mh) + poisson_nll(ya, ma))
    nb_ll = 0.5 * (nb_nll(yh, mh, alpha) + nb_nll(ya, ma, alpha))
    mae = 0.5 * (np.mean(np.abs(yh - mh)) + np.mean(np.abs(ya - ma)))
    return {
        "status": "PASS",
        "candidate_eligible": bool(alpha > 0 and (poisson_ll - nb_ll) >= MIN_IMPROVEMENT),
        "rows": int(len(df)),
        "fit_rows": int(len(fit)),
        "final_oos_rows": int(len(test)),
        "alpha": float(alpha),
        "poisson_nll": float(poisson_ll),
        "negative_binomial_nll": float(nb_ll),
        "delta_nll": float(nb_ll - poisson_ll),
        "mean_score_mae": float(mae),
        "production_auto_promotion": False,
    }


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": 1, "artifact_type": "NEGATIVE_BINOMIAL_SCORE_OOS", "status": "DEFERRED", "production_auto_promotion": False, "leagues": {}}
    for league in ("NPB", "MLB"):
        path = RESULTS / f"{league.lower()}_backtest_results.csv"
        payload["leagues"][league] = evaluate(path) if path.exists() else {"status": "DEFERRED", "reason": "OOS result file missing", "rows": 0}
    payload["status"] = "PASS" if any(v.get("status") == "PASS" for v in payload["leagues"].values()) else "DEFERRED"
    (RESULTS / "negative_binomial_score_oos.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
