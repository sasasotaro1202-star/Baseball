#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Final CI hardening chain for the baseball production backtest."""
from __future__ import annotations
from pathlib import Path
import runpy
import re

P = Path("baseball_backtest.py")
s = P.read_text(encoding="utf-8")

# Quality-first production ceiling: one hour per workflow slice.
# The workflow and engine must agree so the budget cannot be silently defeated.
if "# BASEBALL_RUNTIME_BUDGET_HARDENING_V2" not in s:
    # This patch can run after MLB_SCORE_HILO_PATCH_V2 and/or other runtime
    # patches. Accept either the historical min() form or the MLB max() form,
    # and normalize both to the same one-hour workflow-controlled contract.
    import re
    pat = r'self\.time_budget_sec\s*=\s*(?:min\(float\(os\.getenv\("BASEBALL_TIME_BUDGET_SEC",\s*"1500"\)\),\s*[0-9.]+\)|max\(60\.0,\s*float\(os\.getenv\("BASEBALL_TIME_BUDGET_SEC",\s*"1500"\)\)))\s*(?:#.*)?'
    s, n = re.subn(pat, 'self.time_budget_sec = min(float(os.getenv("BASEBALL_TIME_BUDGET_SEC", "1500")), 3600.0)  # production ceiling: 60:00', s, count=1)
    if n == 0:
        # Fallback for a future equivalent expression: locate the assignment
        # line by its stable symbol and replace only that line.
        lines = s.splitlines()
        found = False
        for i, line in enumerate(lines):
            if "self.time_budget_sec" in line and "BASEBALL_TIME_BUDGET_SEC" in line and "=" in line:
                lines[i] = '        self.time_budget_sec = min(float(os.getenv("BASEBALL_TIME_BUDGET_SEC", "1500")), 3600.0)  # production ceiling: 60:00'
                found = True
                break
        if not found:
            raise SystemExit("[PRODUCTION PATCH] runtime budget assignment not found")
        s = "\n".join(lines) + ("\n" if s.endswith("\n") else "")
    s = s.replace("[HARD STOP] 210-minute production limit reached before processing", "[HARD STOP] 60-minute production limit reached before processing")
    s = s.replace("[HARD STOP] 210-minute production limit reached; skipping remaining leagues", "[HARD STOP] 60-minute production limit reached; skipping remaining leagues")
    s = "# BASEBALL_RUNTIME_BUDGET_HARDENING_V2\n" + s
    P.write_text(s, encoding="utf-8")
    print("[PRODUCTION PATCH] 60-minute runtime budget hardening applied")
else:
    print("[PRODUCTION PATCH] 60-minute runtime budget hardening already applied")

s = P.read_text(encoding="utf-8")
if "# BASEBALL_PRODUCTION_HARDENING_V1" not in s:
    old_version = 'self.checkpoint_version = "npb-massive-resume-v4-100target"'
    if old_version in s:
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

# Apply the complete quality/source/model hardening chain during production.
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
