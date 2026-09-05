#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Final CI hardening for baseball_backtest.py.

The historical OOS engine must simulate a pregame decision state.  Therefore
starter-incomplete samples are rejected instead of silently receiving generic
pitcher priors.  The patch is idempotent and creates a new checkpoint namespace
so results produced under the older contract are never mixed into production.
"""
from __future__ import annotations
from pathlib import Path

P = Path("baseball_backtest.py")
s = P.read_text(encoding="utf-8")

if "# BASEBALL_PRODUCTION_HARDENING_V1" in s:
    print("[PRODUCTION PATCH] V1 already applied")
    raise SystemExit(0)

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
        self.audit.append({
            "type": f"{league.lower()}_starter_coverage",
            "games": int(len(games)),
            "both_starter_rate": starter_rate,
            "required_rate": threshold,
        })
        print(f"[{league} AUDIT] both-starter coverage={starter_rate:.1%} required={threshold:.0%}")
        if starter_rate < threshold:
            raise RuntimeError(
                f"{league} starter coverage too low: {starter_rate:.1%}; "
                f"required >= {threshold:.0%}. Refusing to run a misleading backtest."
            )

'''
s = s[:start] + gate + s[end:]
s = "# BASEBALL_PRODUCTION_HARDENING_V1\n" + s
P.write_text(s, encoding="utf-8")
print("[PRODUCTION PATCH] V1 applied")
