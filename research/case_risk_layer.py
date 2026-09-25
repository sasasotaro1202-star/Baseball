#!/usr/bin/env python3
"""Case-level uncertainty / upset-risk diagnostics.

Research-only by design: this layer scores uncertainty, data insufficiency,
model disagreement, and distribution shift without reversing a prediction.
Any future decision policy must pass chronological OOS and frozen-holdout gates.
"""
from __future__ import annotations

import math
from typing import Iterable, Mapping, Sequence

import numpy as np


def _clip01(x: float) -> float:
    try:
        return float(np.clip(float(x), 0.0, 1.0))
    except Exception:
        return 0.0


def _norm_probs(probs: Sequence[float]) -> np.ndarray:
    p = np.asarray(list(probs), dtype=float).reshape(-1)
    if p.size < 2 or not np.all(np.isfinite(p)):
        raise ValueError("at least two finite probabilities are required")
    p = np.clip(p, 1e-12, None)
    total = float(p.sum())
    if not math.isfinite(total) or total <= 0:
        raise ValueError("probabilities must have positive mass")
    return p / total


def normalized_entropy(probs: Sequence[float]) -> float:
    p = _norm_probs(probs)
    return _clip01(float(-(p * np.log(p)).sum() / math.log(len(p))))


def top_gap_risk(probs: Sequence[float], scale: float = 0.25) -> float:
    p = np.sort(_norm_probs(probs))
    gap = float(p[-1] - p[-2])
    return _clip01(1.0 - gap / max(float(scale), 1e-9))


def _state_risk(value: str, verified=0.0, projected=0.35, unknown=1.0) -> float:
    v = str(value or "").upper()
    if v == "VERIFIED":
        return float(verified)
    if v in {"PROJECTED", "EXPECTED"}:
        return float(projected)
    return float(unknown)


def _count_risk(n: int, full: int) -> float:
    if full <= 0:
        return 0.0
    return _clip01(float(n) / float(full))


def assess_case_risk(
    probs: Sequence[float],
    *,
    disagreement: float = 0.0,
    conformal: float = 0.0,
    drift: float = 0.0,
    feature_drift: float = 0.0,
    output_drift: float = 0.0,
    starter_state: str = "UNKNOWN",
    lineup_state: str = "UNKNOWN",
    weather_state: str = "UNKNOWN",
    roster_events: Iterable[Mapping[str, object]] | None = None,
    rest_travel: Mapping[str, object] | None = None,
    observation_count: int = 0,
    usable_observation_count: int = 0,
) -> dict[str, object]:
    """Return a deterministic, auditable case-risk profile.

    Higher values mean more uncertainty / information fragility. The signal is
    deliberately NOT interpreted as "pick the underdog" or "flip the side".
    """
    p = _norm_probs(probs)
    entropy = normalized_entropy(p)
    gap_risk = top_gap_risk(p)
    model_risk = _clip01(
        0.45 * entropy
        + 0.25 * gap_risk
        + 0.20 * _clip01(disagreement)
        + 0.10 * _clip01(conformal)
    )

    starter_risk = _state_risk(starter_state)
    lineup_risk = _state_risk(lineup_state, projected=0.45)
    weather_risk = _state_risk(weather_state, projected=0.25)
    events = list(roster_events or [])
    roster_risk = _clip01(0.15 * len(events))
    if observation_count > 0:
        missing_risk = _clip01(
            1.0 - float(usable_observation_count) / float(observation_count)
        )
    else:
        missing_risk = 0.0
    data_risk = _clip01(
        0.34 * starter_risk
        + 0.30 * lineup_risk
        + 0.16 * weather_risk
        + 0.10 * roster_risk
        + 0.10 * missing_risk
    )

    drift_risk = _clip01(
        0.50 * _clip01(drift)
        + 0.30 * _clip01(feature_drift)
        + 0.20 * _clip01(output_drift)
    )

    # Situational anomalies are diagnostics, not forced outcome flips.
    rt = dict(rest_travel or {})
    situational_parts = []
    for side in ("home", "away"):
        item = rt.get(side) if isinstance(rt.get(side), Mapping) else {}
        rest_days = float(item.get("rest_days", 30.0) or 30.0)
        games_3d = float(item.get("games_last_3d", 0.0) or 0.0)
        games_7d = float(item.get("games_last_7d", 0.0) or 0.0)
        travel_miles = float(item.get("travel_miles", 0.0) or 0.0)
        short_rest = _clip01((2.0 - rest_days) / 2.0)
        density = _clip01(max(games_3d / 3.0, games_7d / 7.0))
        travel = _clip01(travel_miles / 1000.0)
        situational_parts.append(0.45 * short_rest + 0.35 * density + 0.20 * travel)
    situational_risk = float(max(situational_parts, default=0.0))

    instability_risk = _clip01(
        0.55 * _clip01(disagreement)
        + 0.45 * _clip01(conformal)
    )

    score = _clip01(
        0.34 * model_risk
        + 0.25 * data_risk
        + 0.18 * drift_risk
        + 0.13 * instability_risk
        + 0.10 * situational_risk
    )
    state = "HIGH" if score >= 0.70 else "MEDIUM" if score >= 0.45 else "LOW"

    reasons = []
    if entropy >= 0.70:
        reasons.append("high_probability_entropy")
    if gap_risk >= 0.65:
        reasons.append("small_top_probability_gap")
    if _clip01(disagreement) >= 0.60:
        reasons.append("model_disagreement")
    if _clip01(conformal) >= 0.60:
        reasons.append("wide_conformal_uncertainty")
    if drift_risk >= 0.60:
        reasons.append("distribution_shift")
    if lineup_risk >= 0.60:
        reasons.append("lineup_information_fragile")
    if starter_risk >= 0.60:
        reasons.append("starter_information_fragile")
    if weather_risk >= 0.60:
        reasons.append("weather_information_fragile")
    if missing_risk >= 0.50:
        reasons.append("observation_coverage_gap")
    if events:
        reasons.append("roster_change_signal")
    if situational_risk >= 0.60:
        reasons.append("situational_load_or_travel_anomaly")

    # This is intentionally named as a signal rather than a forced outcome.
    upset_signal = _clip01(
        0.40 * score
        + 0.25 * drift_risk
        + 0.20 * _clip01(disagreement)
        + 0.15 * gap_risk
    )
    return {
        "schema_version": 1,
        "research_only": True,
        "risk_state": state,
        "risk_score": float(score),
        "upset_risk_signal": float(upset_signal),
        "model_risk": float(model_risk),
        "data_risk": float(data_risk),
        "drift_risk": float(drift_risk),
        "instability_risk": float(instability_risk),
        "probability_entropy": float(entropy),
        "top_gap_risk": float(gap_risk),
        "starter_risk": float(starter_risk),
        "lineup_risk": float(lineup_risk),
        "weather_risk": float(weather_risk),
        "missing_observation_risk": float(missing_risk),
        "situational_risk": float(situational_risk),
        "reasons": reasons,
    }
