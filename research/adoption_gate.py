"""Leakage-safe Baseball candidate adoption gate.

Selection and final confirmation are deliberately separated:
Development OOS -> Candidate Lock -> Locked Holdout -> Adoption.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class GatePolicy:
    min_oos_rows: int = 200
    min_relative_improvement: float = 0.01
    max_logloss_regression: float = 0.005
    max_brier_regression: float = 0.005
    max_accuracy_regression: float = 0.005
    require_two_validation_windows: bool = True
    require_calibration_check: bool = True
    require_no_future_target_data: bool = True
    require_reproducible_candidate: bool = True


def candidate_lock(*, development_metrics: Mapping[str, float], candidate_id: str) -> dict:
    """Lock a candidate chosen on Development OOS without evaluating the holdout."""
    if not candidate_id:
        raise ValueError("candidate_id is required")
    rows = int(development_metrics.get("rows", 0))
    return {
        "candidate_id": candidate_id,
        "stage": "candidate_locked",
        "development_rows": rows,
        "development_metrics": dict(development_metrics),
    }


def evaluate_locked_holdout(
    baseline: Mapping[str, float],
    candidate: Mapping[str, float],
    *,
    policy: GatePolicy = GatePolicy(),
    validation_windows: int = 0,
    calibration_ok: bool = False,
    no_future_target_data: bool = False,
    reproducible: bool = False,
) -> dict:
    """Compare baseline and locked candidate on unseen holdout data only."""
    rows = int(candidate.get("rows", 0))
    reasons: list[str] = []
    if rows < policy.min_oos_rows:
        reasons.append("insufficient_locked_holdout_rows")
    if policy.require_two_validation_windows and validation_windows < 2:
        reasons.append("insufficient_validation_windows")
    if policy.require_calibration_check and not calibration_ok:
        reasons.append("calibration_check_failed")
    if policy.require_no_future_target_data and not no_future_target_data:
        reasons.append("future_target_data_not_excluded")
    if policy.require_reproducible_candidate and not reproducible:
        reasons.append("candidate_not_reproducible")

    # Lower is better for LogLoss/Brier; higher is better for Accuracy.
    ll_base = float(baseline.get("LogLoss", float("inf")))
    ll_cand = float(candidate.get("LogLoss", float("inf")))
    br_base = float(baseline.get("Brier", float("inf")))
    br_cand = float(candidate.get("Brier", float("inf")))
    acc_base = float(baseline.get("Accuracy", 0.0))
    acc_cand = float(candidate.get("Accuracy", 0.0))

    ll_improvement = ll_base - ll_cand
    br_improvement = br_base - br_cand
    acc_improvement = acc_cand - acc_base
    relative_ll = ll_improvement / max(abs(ll_base), 1e-12)

    if relative_ll < policy.min_relative_improvement:
        reasons.append("logloss_improvement_below_gate")
    if ll_cand - ll_base > policy.max_logloss_regression:
        reasons.append("logloss_regression")
    if br_cand - br_base > policy.max_brier_regression:
        reasons.append("brier_regression")
    if acc_base - acc_cand > policy.max_accuracy_regression:
        reasons.append("accuracy_regression")

    return {
        "stage": "locked_holdout_evaluated",
        "adopt": not reasons,
        "reasons": reasons,
        "baseline": dict(baseline),
        "candidate": dict(candidate),
        "improvement": {
            "LogLoss": ll_improvement,
            "Brier": br_improvement,
            "Accuracy": acc_improvement,
            "relative_LogLoss": relative_ll,
        },
    }
