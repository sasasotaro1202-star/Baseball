#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Production patch: unlock long OOS budgets and make MLB score/Low-High fields explicit.

Applied at runtime so the patch is idempotent and auditable. The underlying
backtest already computes MLB score lambdas, four score candidates, and Low/High
probabilities; this patch removes the accidental 1500-second global cap and
adds explicit categorical/probability aliases to the persisted prediction rows.
"""
from pathlib import Path

P = Path("baseball_backtest.py")
s = P.read_text(encoding="utf-8")

if "# MLB_SCORE_HILO_PATCH_V1" in s:
    print("[MLB SCORE/HILO PATCH] already applied")
    raise SystemExit(0)

old = 'self.time_budget_sec = min(float(os.getenv("BASEBALL_TIME_BUDGET_SEC", "1500")), 1500.0)  # hard cap: 29:00'
new = 'self.time_budget_sec = max(60.0, float(os.getenv("BASEBALL_TIME_BUDGET_SEC", "1500")))  # workflow controls the budget'
if old not in s:
    raise RuntimeError("expected global time-budget cap not found")
s = s.replace(old, new, 1)

# Make the MLB score/Low-High contract explicit in persisted rows. Existing
# lambda_home/lambda_away + score1..score4 + low/high remain untouched.
needle = '                    "low": low, "high": high,\n                    "actual_home_score": float(r["home_score"]), "actual_away_score": float(r["away_score"]),\n'
replacement = '                    "low": low, "high": high,\n                    "pred_low_prob": low, "pred_high_prob": high,\n                    "pred_low_high": "Low" if low >= 0.5 else "High",\n                    "pred_score1": scores[0][0], "pred_score1_prob": scores[0][1],\n                    "pred_score2": scores[1][0], "pred_score2_prob": scores[1][1],\n                    "pred_score3": scores[2][0], "pred_score3_prob": scores[2][1],\n                    "pred_score4": scores[3][0], "pred_score4_prob": scores[3][1],\n                    "actual_home_score": float(r["home_score"]), "actual_away_score": float(r["away_score"]),\n'
if needle not in s:
    raise RuntimeError("prediction row insertion point not found")
s = s.replace(needle, replacement, 1)

# Add explicit postgame categorical hit fields to every saved row. These are
# derived only from the completed target game's actual score, after prediction.
needle2 = '                    "actual_home_score": float(r["home_score"]), "actual_away_score": float(r["away_score"]),\n                })'
replacement2 = '                    "actual_home_score": float(r["home_score"]), "actual_away_score": float(r["away_score"]),\n                    "actual_low_high": "High" if (float(r["home_score"]) >= 7 or float(r["away_score"]) >= 7) else "Low",\n                    "score_exact_hit": int(any(str(int(float(r["home_score"]))) + "-" + str(int(float(r["away_score"]))) == str(z[0]) for z in scores if str(z[0]) != "その他")),\n                    "low_high_hit": int(("High" if (float(r["home_score"]) >= 7 or float(r["away_score"]) >= 7) else "Low") == ("Low" if low >= 0.5 else "High")),\n                })'
if needle2 not in s:
    raise RuntimeError("actual score insertion point not found")
s = s.replace(needle2, replacement2, 1)

s = "# MLB_SCORE_HILO_PATCH_V1\n" + s
P.write_text(s, encoding="utf-8")
print("[MLB SCORE/HILO PATCH] applied: long-budget control + explicit MLB score/Low-High fields")
