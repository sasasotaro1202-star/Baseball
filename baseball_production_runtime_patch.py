#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Final CI hardening chain for the baseball production backtest."""
from __future__ import annotations
from pathlib import Path
import runpy

P = Path("baseball_backtest.py")
s = P.read_text(encoding="utf-8")

# The engine historically capped BASEBALL_TIME_BUDGET_SEC at 1500 seconds,
# which silently defeated the quality-first production workflow's 12600-second
# budget. Lift that implementation cap to the workflow ceiling while retaining
# an explicit upper bound against accidental runaway local runs.
if "# BASEBALL_RUNTIME_BUDGET_HARDENING_V1" not in s:
    old = 'self.time_budget_sec = min(float(os.getenv("BASEBALL_TIME_BUDGET_SEC", "1500")), 1500.0)  # hard cap: 29:00'
    new = 'self.time_budget_sec = min(float(os.getenv("BASEBALL_TIME_BUDGET_SEC", "1500")), 12600.0)  # production ceiling: 210:00'
    if old not in s:
        raise SystemExit("[PRODUCTION PATCH] runtime budget anchor not found")
    s = s.replace(old, new, 1)
    s = "# BASEBALL_RUNTIME_BUDGET_HARDENING_V1\n" + s
    P.write_text(s, encoding="utf-8")
    print("[PRODUCTION PATCH] runtime budget hardening applied")
else:
    print("[PRODUCTION PATCH] runtime budget hardening already applied")

# Keep operator-facing hard-stop diagnostics consistent with the actual
# production ceiling. This is correctness/observability, not a model change.
s = P.read_text(encoding="utf-8")
s = s.replace("[HARD STOP] 30-minute limit reached before processing", "[HARD STOP] 210-minute production limit reached before processing")
s = s.replace("[HARD STOP] 30-minute limit reached; skipping remaining leagues", "[HARD STOP] 210-minute production limit reached; skipping remaining leagues")
P.write_text(s, encoding="utf-8")

if "# BASEBALL_PRODUCTION_HARDENING_V1" not in s:
    old_version = 'self.checkpoint_version = "npb-massive-resume-v4-100target"'
    s = s.replace(old_version, 'self.checkpoint_version = "baseball-production-v1-quality-gated"')
    anchor = '''        # Data-quality gates: the backtest must not silently run on a tiny
        # or starter-free sample.
'''
    start = s.find(anchor)
    end = s.find('        X, y, meta = self.build_features(games)', start)
    if start < 0 or end < 0:
        raise SystemExit("[PRODUCTION PATCH] expected walk-forward gate block not found")
    gate = '''        # STRICT STARTER COVERAGE GATE
        # A historical backtest is a simulation of the pregame decision state.
        # Unknown starters must not be silently replaced by league priors.
        # NPB requires >=70%; MLB requires >=90% for the production contract.
        starter_rate = float(
            ((games["home_starter"].fillna("").astype(str).str.strip().str.len() > 0) &
             (games["away_starter"].fillna("").astype(str).str.strip().str.len() > 0)).mean()
        )
        threshold = 0.90 if league == "MLB" else 0.70
        self.audit.append({"type": f"{league.lower()}_starter_coverage", "games": int(len(games)), "both_starter_rate": starter_rate, "required_rate": threshold})
        print(f"[{league} AUDIT] both-starter coverage={starter_rate:.1%} required={threshold:.0%}")
        if starter_rate < threshold:
            raise RuntimeError(f"{league} starter coverage too low: {starter_rate:.1%}; required >= {threshold:.0%}. Refusing to run a misleading backtest.")

'''
    s = s[:start] + gate + s[end:]
    s = "# BASEBALL_PRODUCTION_HARDENING_V1\n" + s
    P.write_text(s, encoding="utf-8")
    print("[PRODUCTION PATCH] V1 applied")
else:
    print("[PRODUCTION PATCH] V1 already applied")

for patch in (
    "npb_official_schedule_patch.py",
    "npb_quality_runtime_patch.py",
    "baseball_mlb_score_hilo_patch.py",
    "baseball_quality_runtime_patch.py",
):
    p = Path(patch)
    if not p.exists():
        raise SystemExit(f"[PRODUCTION PATCH] required patch missing: {patch}")
    runpy.run_path(str(p), run_name="__main__")
