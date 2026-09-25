"""Research-only detector for recurring vs emergent pregame drift.

The detector is target-free: it consumes only historical pre-prediction numeric
windows and the current pre-prediction window. It does not alter outcome classes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


_EPS = 1e-12


@dataclass(frozen=True)
class DriftTypeResult:
    recurrence_score: float
    novelty_score: float
    nearest_distance: float
    baseline_distance: float
    drift_type: str


def _finite_matrix(x: np.ndarray) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    if a.ndim != 2 or a.shape[0] < 2 or a.shape[1] == 0:
        raise ValueError("windows must be 2-D with >=2 rows and >=1 feature")
    if not np.all(np.isfinite(a)):
        raise ValueError("windows contain non-finite values")
    return a


def _summary_embedding(window: np.ndarray) -> np.ndarray:
    a = _finite_matrix(window)
    mean = a.mean(axis=0)
    sd = a.std(axis=0, ddof=1)
    return np.concatenate([mean, np.log(np.maximum(sd, _EPS))])


def _distance(a: np.ndarray, b: np.ndarray) -> float:
    x = _summary_embedding(a)
    y = _summary_embedding(b)
    scale = np.maximum(np.abs(x), 1.0)
    raw = np.mean(np.abs(x - y) / scale)
    return float(max(raw, 0.0))


def classify_drift(
    historical_windows: list[np.ndarray],
    current_window: np.ndarray,
    *,
    recurrence_scale: float = 0.35,
    emergent_threshold: float = 0.70,
    recurring_threshold: float = 0.65,
) -> DriftTypeResult:
    """Classify the current pregame state as recurring, stable, or emergent.

    historical_windows must contain only windows strictly earlier than the
    current window. The nearest historical distance is converted to a bounded
    recurrence score. A current-vs-baseline distance is also reported for
    diagnostics; it is not used as a target label.
    """
    if not historical_windows:
        return DriftTypeResult(
            recurrence_score=0.0,
            novelty_score=1.0,
            nearest_distance=float("inf"),
            baseline_distance=float("inf"),
            drift_type="UNKNOWN",
        )

    cur = _finite_matrix(current_window)
    distances = [_distance(w, cur) for w in historical_windows]
    nearest = float(min(distances))

    scale = max(float(recurrence_scale), 1e-6)
    recurrence = float(np.clip(np.exp(-nearest / scale), 0.0, 1.0))
    novelty = float(np.clip(1.0 - recurrence, 0.0, 1.0))

    # The immediately preceding historical window is a local baseline.
    baseline = float(distances[-1])

    if novelty >= float(np.clip(emergent_threshold, 0.0, 1.0)):
        drift_type = "EMERGENT"
    elif recurrence >= float(np.clip(recurring_threshold, 0.0, 1.0)):
        drift_type = "RECURRING"
    else:
        drift_type = "STABLE_OR_TRANSITION"

    return DriftTypeResult(
        recurrence_score=recurrence,
        novelty_score=novelty,
        nearest_distance=nearest,
        baseline_distance=baseline,
        drift_type=drift_type,
    )


def guard_routing_weights(
    proposed_weights: np.ndarray,
    previous_weights: np.ndarray | None,
    *,
    recurrence_score: float,
    novelty_score: float,
    uncertainty: float = 0.0,
    max_change_emergent: float = 0.12,
) -> np.ndarray:
    """Limit routing movement for novel/high-uncertainty cases.

    This is a convex safeguard, not a model-selection rule. Recurring states
    retain more of the proposed routing; emergent states retain more of the
    previously validated weights. Missing previous weights use the proposed
    weights unchanged, preserving a neutral startup path.
    """
    w = np.asarray(proposed_weights, dtype=float).reshape(-1)
    if w.size < 2 or not np.all(np.isfinite(w)) or np.any(w < 0):
        raise ValueError("proposed_weights are invalid")
    w = w / max(float(w.sum()), _EPS)

    if previous_weights is None:
        return w

    prev = np.asarray(previous_weights, dtype=float).reshape(-1)
    if len(prev) != len(w) or not np.all(np.isfinite(prev)) or np.any(prev < 0):
        raise ValueError("previous_weights are invalid")
    prev = prev / max(float(prev.sum()), _EPS)

    recurrence = float(np.clip(recurrence_score, 0.0, 1.0))
    novelty = float(np.clip(novelty_score, 0.0, 1.0))
    unc = float(np.clip(uncertainty, 0.0, 1.0))

    # High novelty + uncertainty => conservative movement. Recurrence gives
    # back some adaptation budget because an analogous state has appeared before.
    adapt = (0.15 + 0.55 * recurrence) * (1.0 - 0.60 * novelty * unc)
    adapt = float(np.clip(adapt, 0.0, 1.0))

    # Hard ceiling on per-call movement for emergent states.
    if novelty >= 0.70 and unc >= 0.60:
        adapt = min(adapt, float(np.clip(max_change_emergent, 0.0, 1.0)))

    out = (1.0 - adapt) * prev + adapt * w
    out = np.maximum(out, 0.0)
    return out / max(float(out.sum()), _EPS)
