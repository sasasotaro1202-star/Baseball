#!/usr/bin/env python3
"""Research-only acceptance gate for dynamic expert routing.

The gate consumes routing_oos_replay.json and requires the routed+recalibrated
candidate to improve on the incumbent baseline in two non-overlapping late
chronological OOS windows. It never changes production state.
"""
from __future__ import annotations

import json
from pathlib import Path

RESULTS = Path("results")
INPUT = RESULTS / "routing_oos_replay.json"
OUTPUT = RESULTS / "routing_acceptance_gate.json"

MIN_WINDOW_ROWS = 50
MIN_LOGLOSS_IMPROVEMENT = 0.0005
MIN_BRIER_IMPROVEMENT = 0.00025
MAX_ACCURACY_REGRESSION = 0.005


def gate_window(window: dict) -> dict:
    base = window.get("baseline") or {}
    routed = window.get("routed_recalibrated") or {}
    rows = int(window.get("rows") or 0)
    dll = float(routed.get("logloss", float("nan")) - base.get("logloss", float("nan")))
    db = float(routed.get("brier", float("nan")) - base.get("brier", float("nan")))
    da = float(routed.get("accuracy", float("nan")) - base.get("accuracy", float("nan")))
    passed = (
        rows >= MIN_WINDOW_ROWS
        and dll <= -MIN_LOGLOSS_IMPROVEMENT
        and db <= -MIN_BRIER_IMPROVEMENT
        and da >= -MAX_ACCURACY_REGRESSION
    )
    return {
        "rows": rows,
        "delta_logloss": dll,
        "delta_brier": db,
        "delta_accuracy": da,
        "passed": bool(passed),
    }


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    if not INPUT.exists() or INPUT.stat().st_size == 0:
        payload = {
            "status": "DEFERRED",
            "candidate_eligible": False,
            "reason": "routing OOS replay artifact is missing",
        }
        OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(payload, ensure_ascii=False))
        return 0

    replay = json.loads(INPUT.read_text(encoding="utf-8"))
    checked = []
    for artifact in replay.get("results", []):
        if artifact.get("status") != "PASS":
            checked.append({
                "checkpoint": artifact.get("checkpoint"),
                "status": "DEFERRED",
                "reason": artifact.get("reason", "replay artifact not usable"),
            })
            continue
        windows = artifact.get("late_oos_windows") or {}
        ordered = [windows[k] for k in sorted(windows) if isinstance(windows[k], dict)]
        results = [gate_window(w) for w in ordered[:2]]
        eligible = len(results) >= 2 and all(r["passed"] for r in results)
        checked.append({
            "checkpoint": artifact.get("checkpoint"),
            "league": artifact.get("league"),
            "status": "PASS" if results else "DEFERRED",
            "candidate_eligible": bool(eligible),
            "windows_checked": results,
            "policy": {
                "requires_two_non_overlapping_late_oos_windows": True,
                "min_logloss_improvement": MIN_LOGLOSS_IMPROVEMENT,
                "min_brier_improvement": MIN_BRIER_IMPROVEMENT,
                "max_accuracy_regression": MAX_ACCURACY_REGRESSION,
                "production_auto_promotion": False,
            },
        })

    usable = [x for x in checked if x.get("status") == "PASS"]
    eligible_count = sum(bool(x.get("candidate_eligible")) for x in usable)
    payload = {
        "schema_version": 1,
        "status": "PASS" if usable else "DEFERRED",
        "candidate_eligible_count": eligible_count,
        "results": checked,
        "production_auto_promotion": False,
        "promotion_requires_external_frozen_holdout_and_integrity_gates": True,
    }
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
