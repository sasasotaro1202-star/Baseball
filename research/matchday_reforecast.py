#!/usr/bin/env python3
"""Research-only Matchday Intelligence -> bounded reforecast controller.

No context adjustment is hard-coded here.  The controller consumes an OOS-fitted
effect artifact whose coefficients were learned on historical, PIT-safe context
deltas.  Missing/stale/conflicted observations are ignored, not guessed.

Pipeline:
baseline probability
 -> PIT-safe context selection
 -> bounded learned context delta
 -> drift/uncertainty expert routing
 -> final recalibration

Production promotion remains external and requires frozen-holdout/integrity gates.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from research.drift_uncertainty_routing import (
    AdaptiveTemperatureCalibrator,
    RoutingConfig,
    mix_expert_probabilities,
    route_experts,
)
from research.matchday_intelligence import ContextState, Observation, is_pit_safe


RESULTS = Path("results")
EFFECTS = RESULTS / "matchday_effects.json"

EPS = 1e-12


@dataclass(frozen=True)
class EffectSpec:
    """Multiclass additive probability-effect approximation in logit space."""
    key: str
    class_index: int
    coefficient: float
    max_abs_logit: float


@dataclass(frozen=True)
class ReforecastResult:
    baseline: np.ndarray
    context_adjusted: np.ndarray
    final: np.ndarray
    applied_events: List[str]
    skipped_events: List[Dict[str, str]]
    status: str
    reason: Optional[str] = None


def _normalize(p: np.ndarray) -> np.ndarray:
    x = np.asarray(p, dtype=float).reshape(-1)
    if len(x) < 2 or not np.all(np.isfinite(x)):
        raise ValueError("probability vector invalid")
    x = np.clip(x, EPS, 1.0)
    return x / max(float(x.sum()), EPS)


def _softmax(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    z = z - np.max(z)
    e = np.exp(np.clip(z, -40.0, 40.0))
    return e / max(float(e.sum()), EPS)


def load_effects(path: Path = EFFECTS) -> Tuple[List[EffectSpec], Dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError("matchday effect artifact is missing")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "PASS" or payload.get("artifact_type") != "OOS_LEARNED_MATCHDAY_EFFECTS":
        raise ValueError("matchday effect artifact is not promotable")
    specs = []
    for raw in payload.get("effects", []):
        coef = float(raw.get("coefficient"))
        cap = float(raw.get("max_abs_logit", 0.15))
        if not np.isfinite(coef) or not np.isfinite(cap) or cap <= 0:
            raise ValueError("invalid matchday effect coefficient")
        specs.append(
            EffectSpec(
                key=str(raw["key"]),
                class_index=int(raw["class_index"]),
                coefficient=float(np.clip(coef, -cap, cap)),
                max_abs_logit=float(cap),
            )
        )
    alpha = payload.get("fusion_alpha")
    if alpha is None or not np.isfinite(float(alpha)):
        raise ValueError("learned fusion alpha is missing")
    payload["fusion_alpha"] = float(np.clip(float(alpha), 0.0, 1.0))
    if not specs:
        raise ValueError("no learned matchday effects")
    return specs, payload


def _event_value(event: str, observation: Observation) -> float:
    value = observation.value
    if isinstance(value, (int, float)) and np.isfinite(float(value)):
        return float(value)
    if isinstance(value, dict):
        raw = value.get(event) or value.get("magnitude") or value.get("delta")
        if isinstance(raw, (int, float)) and np.isfinite(float(raw)):
            return float(raw)
    # Presence-only events use 1.0. The learned coefficient decides direction
    # and magnitude; this is not a hand-coded performance assumption.
    return 1.0


def apply_context_effects(
    baseline: np.ndarray,
    observations: Iterable[Observation],
    effects: Iterable[EffectSpec],
) -> Tuple[np.ndarray, List[str], List[Dict[str, str]]]:
    p = _normalize(baseline)
    logp = np.log(np.clip(p, EPS, 1.0))
    applied: List[str] = []
    skipped: List[Dict[str, str]] = []

    specs = list(effects)
    by_key: Dict[str, List[EffectSpec]] = {}
    for spec in specs:
        by_key.setdefault(spec.key, []).append(spec)
    for obs in observations:
        if not is_pit_safe(obs):
            skipped.append({"snapshot_id": obs.snapshot_id, "reason": "PIT_FAIL"})
            continue
        state = str(obs.state).upper()
        if state not in {ContextState.VERIFIED.value, ContextState.PROJECTED.value}:
            skipped.append({"snapshot_id": obs.snapshot_id, "reason": f"state={state}"})
            continue
        candidates = [f"ctx_{obs.value}", f"{obs.kind}:{obs.value}", str(obs.value), obs.kind]
        key = next((k for k in candidates if k in by_key), None)
        if key is None:
            skipped.append({"snapshot_id": obs.snapshot_id, "reason": "NO_LEARNED_EFFECT"})
            continue
        magnitude = 1.0 if isinstance(obs.value, str) else float(np.clip(_event_value(key, obs), -3.0, 3.0))
        for spec in by_key[key]:
            delta = float(np.clip(spec.coefficient * magnitude, -spec.max_abs_logit, spec.max_abs_logit))
            idx = int(spec.class_index)
            if not (0 <= idx < len(logp)):
                skipped.append({"snapshot_id": obs.snapshot_id, "reason": "CLASS_INDEX_OUT_OF_RANGE"})
                continue
            logp[idx] += delta
            applied.append(f"{key}:class_{idx}")
    return _softmax(logp), applied, skipped


def reforecast(
    baseline: np.ndarray,
    expert_probs: np.ndarray,
    history_logloss: np.ndarray,
    observations: Iterable[Observation],
    previous_weights: Optional[np.ndarray] = None,
    drift_score: float = 0.0,
    conformal_uncertainty: float = 0.0,
    calibrator: Optional[AdaptiveTemperatureCalibrator] = None,
    config: RoutingConfig = RoutingConfig(),
    effect_path: Path = EFFECTS,
) -> ReforecastResult:
    try:
        effects, effect_payload = load_effects(effect_path)
    except Exception as exc:
        return ReforecastResult(
            baseline=_normalize(baseline),
            context_adjusted=_normalize(baseline),
            final=_normalize(baseline),
            applied_events=[],
            skipped_events=[],
            status="DEFERRED",
            reason=str(exc),
        )

    context_adjusted, applied, skipped = apply_context_effects(baseline, observations, effects)
    expert_p = np.asarray(expert_probs, dtype=float)
    routed = route_experts(
        expert_probs=expert_p,
        history_logloss=np.asarray(history_logloss, dtype=float),
        drift_score=float(np.clip(drift_score, 0.0, 1.0)),
        previous_weights=previous_weights,
        conformal_uncertainty=conformal_uncertainty,
        config=config,
    )
    # Context adjustment is an overlay, not a replacement of the expert ensemble.
    mixed = mix_expert_probabilities(expert_p, routed.weights)
    alpha = float(np.clip(effect_payload["fusion_alpha"], 0.0, 1.0))
    mixed = _normalize((1.0 - alpha) * mixed + alpha * context_adjusted)
    cal = calibrator or AdaptiveTemperatureCalibrator(config=config)
    final = cal.predict(mixed)
    return ReforecastResult(
        baseline=_normalize(baseline),
        context_adjusted=context_adjusted,
        final=_normalize(final),
        applied_events=applied,
        skipped_events=skipped,
        status="PASS",
    )


def self_test() -> Dict[str, Any]:
    """Deterministic smoke test. No effect artifact => fail-closed DEFERRED."""
    baseline = np.array([0.55, 0.25, 0.20])
    experts = np.array([
        [0.58, 0.23, 0.19],
        [0.52, 0.28, 0.20],
    ])
    history = np.ones((30, 2), dtype=float)
    obs = []
    result = reforecast(baseline, experts, history, obs)
    assert result.status == "DEFERRED"
    assert np.isclose(result.final.sum(), 1.0)
    return {"status": "PASS", "controller_status": result.status}


if __name__ == "__main__":
    print(json.dumps(self_test(), ensure_ascii=False, sort_keys=True))
