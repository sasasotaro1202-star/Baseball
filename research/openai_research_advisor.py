#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate a deterministic, local research advisory from prior OOS state only.

No external API, secret, or paid service is used. The output is advisory-only.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "research_state.json"
PLAN = ROOT / "research_plan.json"
OUT = ROOT / "research_ai_advice.json"


def load_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default
    except Exception:
        return default


def n(v):
    try:
        x = float(v)
        return x if x == x and abs(x) != float("inf") else None
    except Exception:
        return None


def summary(league_state):
    rows = league_state.get("model_leaderboard") or []
    if not rows:
        return "no model leaderboard is currently available"
    top = rows[0]
    parts = [f"top={top.get('model', 'unknown')}"]
    for key in ("LogLoss", "Brier", "Accuracy"):
        value = n(top.get(key))
        if value is not None:
            parts.append(f"{key}={value:.6f}" if key != "Accuracy" else f"{key}={value:.4f}")
    return ", ".join(parts)


def build_advice(state, plan):
    focus = state.get("focus") or plan.get("priority") or {}
    gate = state.get("promotion_gate") or plan.get("promotion_gate") or {}
    leagues = state.get("leagues") or {}
    return {
        "focus": focus,
        "evidence_from_current_state": {
            "NPB": summary(leagues.get("NPB") or {}),
            "MLB": summary(leagues.get("MLB") or {}),
            "source_policy": "prior_rows_only",
            "missing_data_is_not_invented": True,
        },
        "top_research_hypotheses": [
            {
                "name": "chronological ensemble stability",
                "test": "Compare current weighted ensemble with frozen and recency-weighted variants on two untouched chronological windows.",
                "guard": "Require OOS improvement in LogLoss and Brier with no material Accuracy regression.",
            },
            {
                "name": "starter quality interactions",
                "test": "Ablate pregame starter quality gaps and uncertainty terms by coverage tier.",
                "guard": "Reject any candidate lacking defensible pregame provenance.",
            },
            {
                "name": "bullpen and environment interactions",
                "test": "Test bullpen fatigue, park, weather and run-environment interactions independently before combining them.",
                "guard": "Never replace missing historical observations with future or target-game data.",
            },
            {
                "name": "score-to-win coherence",
                "test": "Measure disagreement between score-distribution and win-probability outputs by context.",
                "guard": "Use only relationships that survive both chronological validation windows and calibration checks.",
            },
        ],
        "recommended_oos_tests": [
            "chronological two-window OOS comparison",
            "LogLoss/Brier calibration check",
            "feature ablation without future target rows",
            "robustness by starter coverage, home/away and scoring environment",
        ],
        "failure_modes_to_check": [
            "timestamp or provenance ambiguity",
            "starter/lineup contamination",
            "single-window overfitting",
            "production-state writes before all gates pass",
        ],
        "promotion_conditions": gate,
    }


def main():
    state = load_json(STATE, {})
    plan = load_json(PLAN, {})
    advice = build_advice(state, plan)
    OUT.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "model": "local-deterministic-advisor-v1",
                "source_policy": "prior_rows_only",
                "advisory_only": True,
                "network_used": False,
                "advice": advice,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(advice, ensure_ascii=False))


if __name__ == "__main__":
    main()
