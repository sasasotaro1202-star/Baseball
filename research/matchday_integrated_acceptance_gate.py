#!/usr/bin/env python3
"""Integrated acceptance gate for Matchday Intelligence + dynamic routing.

A candidate is eligible only when:
1) dynamic routing has cleared its two-window OOS gate and high-state checks;
2) Matchday effects have cleared their chronological OOS gate;
3) canonical forward Matchday replay also improves LogLoss/Brier with bounded
   Accuracy/ECE change.

This file never promotes production state.
"""
from __future__ import annotations

import json
from pathlib import Path

RESULTS = Path("results")
ROUTING = RESULTS / "routing_acceptance_gate.json"
EFFECTS = RESULTS / "matchday_effects.json"
REPLAY = RESULTS / "matchday_replay.json"
OUTPUT = RESULTS / "matchday_integrated_acceptance_gate.json"

MIN_LL = 0.0005
MIN_BRIER = 0.00025
MAX_ACC_REG = 0.005
MAX_ECE_REG = 0.010


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    missing = [str(p) for p in (ROUTING, EFFECTS, REPLAY) if not p.exists() or p.stat().st_size == 0]
    if missing:
        payload = {
            "schema_version": 1,
            "status": "DEFERRED",
            "candidate_eligible": False,
            "reason": "required gate artifact missing",
            "missing": missing,
            "production_auto_promotion": False,
        }
        OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(payload, ensure_ascii=False))
        return 0

    try:
        routing = json.loads(ROUTING.read_text(encoding="utf-8"))
        effects = json.loads(EFFECTS.read_text(encoding="utf-8"))
        replay = json.loads(REPLAY.read_text(encoding="utf-8"))
    except Exception as exc:
        payload = {
            "schema_version": 1,
            "status": "DEFERRED",
            "candidate_eligible": False,
            "reason": f"gate artifact parse failed: {exc}",
            "production_auto_promotion": False,
        }
        OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(payload, ensure_ascii=False))
        return 0

    routing_ok = any(
        bool(x.get("candidate_eligible"))
        for x in routing.get("results", [])
        if isinstance(x, dict)
    )
    effect_ok = bool(effects.get("candidate_eligible")) and effects.get("status") == "PASS"

    replay_delta = replay.get("delta") or {}
    replay_ok = (
        replay.get("status") == "PASS"
        and float(replay_delta.get("delta_logloss", 0.0)) <= -MIN_LL
        and float(replay_delta.get("delta_brier", 0.0)) <= -MIN_BRIER
        and float(replay_delta.get("delta_accuracy", 0.0)) >= -MAX_ACC_REG
        and float(replay_delta.get("delta_ece", 0.0)) <= MAX_ECE_REG
    )

    eligible = routing_ok and effect_ok and replay_ok
    payload = {
        "schema_version": 1,
        "status": "PASS" if replay.get("status") == "PASS" else "DEFERRED",
        "candidate_eligible": bool(eligible),
        "routing_gate": routing_ok,
        "effect_gate": effect_ok,
        "forward_replay_gate": replay_ok,
        "forward_replay_delta": replay_delta,
        "policy": {
            "min_logloss_improvement": MIN_LL,
            "min_brier_improvement": MIN_BRIER,
            "max_accuracy_regression": MAX_ACC_REG,
            "max_ece_regression": MAX_ECE_REG,
            "production_auto_promotion": False,
            "requires_frozen_holdout": True,
            "requires_integrity_gate": True,
        },
    }
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
