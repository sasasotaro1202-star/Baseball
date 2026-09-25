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
MAX_ECE_REGRESSION = 0.010
MAX_HIGH_STATE_LOGLOSS_REGRESSION = 0.010
MIN_LOCAL_LOGLOSS_IMPROVEMENT = 0.0005
MIN_LOCAL_BRIER_IMPROVEMENT = 0.00025
MAX_LOCAL_ACCURACY_REGRESSION = 0.005
MAX_LOCAL_ECE_REGRESSION = 0.010


def gate_window(window: dict) -> dict:
    base = window.get("baseline") or {}
    routed = window.get("routed_recalibrated") or {}
    rows = int(window.get("rows") or 0)
    dll = float(routed.get("logloss", float("nan")) - base.get("logloss", float("nan")))
    db = float(routed.get("brier", float("nan")) - base.get("brier", float("nan")))
    da = float(routed.get("accuracy", float("nan")) - base.get("accuracy", float("nan")))
    de = float(routed.get("ece", float("nan")) - base.get("ece", float("nan")))
    passed = (
        rows >= MIN_WINDOW_ROWS
        and dll <= -MIN_LOGLOSS_IMPROVEMENT
        and db <= -MIN_BRIER_IMPROVEMENT
        and da >= -MAX_ACCURACY_REGRESSION
        and de <= MAX_ECE_REGRESSION
    )
    return {
        "rows": rows,
        "delta_logloss": dll,
        "delta_brier": db,
        "delta_accuracy": da,
        "delta_ece": de,
        "passed": bool(passed),
    }



def gate_local_window(window: dict) -> dict:
    routed = window.get("routed_recalibrated") or {}
    local = window.get("local_competence_shadow") or {}
    rows = int(window.get("rows") or 0)
    dll = float(local.get("logloss", float("nan")) - routed.get("logloss", float("nan")))
    db = float(local.get("brier", float("nan")) - routed.get("brier", float("nan")))
    da = float(local.get("accuracy", float("nan")) - routed.get("accuracy", float("nan")))
    de = float(local.get("ece", float("nan")) - routed.get("ece", float("nan")))
    passed = (
        rows >= MIN_WINDOW_ROWS
        and dll <= -MIN_LOCAL_LOGLOSS_IMPROVEMENT
        and db <= -MIN_LOCAL_BRIER_IMPROVEMENT
        and da >= -MAX_LOCAL_ACCURACY_REGRESSION
        and de <= MAX_LOCAL_ECE_REGRESSION
    )
    return {
        "rows": rows,
        "delta_logloss_vs_global_routing": dll,
        "delta_brier_vs_global_routing": db,
        "delta_accuracy_vs_global_routing": da,
        "delta_ece_vs_global_routing": de,
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
        local_windows = [
            gate_local_window(w)
            for w in ordered[:2]
            if isinstance(w.get("local_competence_shadow"), dict)
        ]
        local_vs_base_safe = all(
            float((w.get("local_competence_shadow") or {}).get("logloss", float("inf")))
            <= float((w.get("baseline") or {}).get("logloss", float("inf"))) + 0.010
            for w in ordered[:2]
        )
        diag = artifact.get("diagnostics") or {}
        high_state_values = [
            float(diag[k])
            for k in ("high_drift_score_delta_logloss", "high_uncertainty_delta_logloss")
            if k in diag
        ]
        high_state_safe = all(v <= MAX_HIGH_STATE_LOGLOSS_REGRESSION for v in high_state_values)
        eligible = len(results) >= 2 and all(r["passed"] for r in results) and high_state_safe
        local_eligible = (
            len(local_windows) >= 2
            and all(r["passed"] for r in local_windows)
            and local_vs_base_safe
        )
        checked.append({
            "checkpoint": artifact.get("checkpoint"),
            "league": artifact.get("league"),
            "status": "PASS" if results else "DEFERRED",
            "candidate_eligible": bool(eligible),
            "windows_checked": results,
            "local_competence_windows_checked": local_windows,
            "local_competence_candidate_eligible": bool(local_eligible),
            "local_competence_safe_vs_global_and_baseline": bool(local_vs_base_safe),
            "high_state_logloss_deltas": high_state_values,
            "high_state_safe": bool(high_state_safe),
            "policy": {
                "requires_two_non_overlapping_late_oos_windows": True,
                "min_logloss_improvement": MIN_LOGLOSS_IMPROVEMENT,
                "min_brier_improvement": MIN_BRIER_IMPROVEMENT,
                "max_accuracy_regression": MAX_ACCURACY_REGRESSION,
                "max_ece_regression": MAX_ECE_REGRESSION,
                "max_high_state_logloss_regression": MAX_HIGH_STATE_LOGLOSS_REGRESSION,
                "production_auto_promotion": False,
            },
        })

    usable = [x for x in checked if x.get("status") == "PASS"]
    eligible_count = sum(bool(x.get("candidate_eligible")) for x in usable)
    payload = {
        "schema_version": 1,
        "status": "PASS" if usable else "DEFERRED",
        "candidate_eligible_count": eligible_count,
        "local_competence_candidate_eligible_count": sum(
            bool(x.get("local_competence_candidate_eligible")) for x in usable
        ),
        "results": checked,
        "production_auto_promotion": False,
        "promotion_requires_external_frozen_holdout_and_integrity_gates": True,
    }
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
