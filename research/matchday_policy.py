#!/usr/bin/env python3
"""Research-only bounded Matchday Policy Engine.

Purpose:
- Keep last-minute context separate from the historical/base model.
- Apply only coefficients that were produced by chronological OOS learning and
  explicitly marked eligible by the routing/matchday acceptance gates.
- Missing, stale, conflicting or weakly evidenced context never creates a
  directional adjustment.
- Final probabilities are always renormalized and remain bounded.

This module is safe to import from a production predictor because an absent or
ineligible policy file is an identity transform.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np


DEFAULT_POLICY = Path("results/matchday_policy.json")
EPS = 1e-12


@dataclass(frozen=True)
class PolicyDecision:
    probabilities: np.ndarray
    applied_effects: Dict[str, float]
    state: str
    eligible: bool
    reason: str


def _prob_vector(probabilities: np.ndarray) -> np.ndarray:
    p = np.asarray(probabilities, dtype=float).reshape(-1)
    if len(p) < 2 or not np.all(np.isfinite(p)) or np.any(p < 0):
        raise ValueError("probabilities are invalid")
    p = np.clip(p, EPS, 1.0)
    return p / max(float(p.sum()), EPS)


def _logit_binary(p: float) -> float:
    q = float(np.clip(p, 1e-6, 1.0 - 1e-6))
    return math.log(q / (1.0 - q))


def _sigmoid(x: float) -> float:
    x = float(np.clip(x, -30.0, 30.0))
    return 1.0 / (1.0 + math.exp(-x))


def load_policy(path: Path = DEFAULT_POLICY) -> Dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        return {
            "status": "DEFERRED",
            "eligible": False,
            "reason": "validated matchday policy is absent",
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "status": "DEFERRED",
            "eligible": False,
            "reason": f"policy parse failed: {exc}",
        }
    if payload.get("status") != "PASS" or payload.get("eligible") is not True:
        return {
            "status": "DEFERRED",
            "eligible": False,
            "reason": payload.get("reason", "policy is not eligible"),
        }
    return payload


def _state_ok(context: Mapping[str, Any], allowed=("VERIFIED", "PROJECTED")) -> bool:
    state = str(context.get("state", "UNKNOWN")).upper()
    return state in set(allowed)


def apply_matchday_policy(
    baseline_probabilities: np.ndarray,
    context: Mapping[str, Mapping[str, Any]],
    *,
    policy: Optional[Mapping[str, Any]] = None,
) -> PolicyDecision:
    """Apply learned bounded context residuals, identity-transform by default.

    For NPB the probabilities are expected to be [home, draw, away].
    Binary inputs [home, away] are also supported.
    Coefficients are interpreted as additive log-odds shifts for the home side
    versus away and are multiplied by evidence confidence/state multipliers.
    """
    base = _prob_vector(baseline_probabilities)
    payload = dict(policy or {})
    if not payload.get("eligible", False):
        return PolicyDecision(base, {}, "DEFERRED", False, payload.get("reason", "no eligible policy"))

    coefficients = payload.get("effects") or {}
    fusion_alpha = float(np.clip(payload.get("fusion_alpha", 1.0), 0.0, 1.0))
    if not isinstance(coefficients, (dict, list)):
        return PolicyDecision(base, {}, "DEFERRED", False, "policy effects are invalid")

    effect_rows = []
    if isinstance(coefficients, dict):
        for key, cfg in coefficients.items():
            item = dict(cfg or {})
            item["key"] = str(key)
            effect_rows.append(item)
    else:
        effect_rows = [dict(x) for x in coefficients if isinstance(x, Mapping)]

    if len(base) == 2:
        home = float(base[0])
        away = float(base[1])
        baseline_logit = _logit_binary(home)
        total_shift = 0.0
        applied: Dict[str, float] = {}
        for cfg in effect_rows:
            key = str(cfg.get("key", ""))
            obs = context.get(key) or {}
            if not _state_ok(obs):
                continue
            coef = float(cfg.get("coef", 0.0))
            cap = abs(float(cfg.get("cap", 0.0)))
            confidence = float(np.clip(obs.get("confidence", 1.0), 0.0, 1.0))
            shift = float(np.clip(coef * confidence, -cap, cap))
            if abs(shift) > 0:
                total_shift += shift
                applied[key] = applied.get(key, 0.0) + shift
        total_shift = float(np.clip(total_shift, -0.65, 0.65))
        ph = _sigmoid(baseline_logit + total_shift)
        out = np.array([ph, 1.0 - ph], dtype=float)
        out = _prob_vector((1.0 - fusion_alpha) * base + fusion_alpha * out)
        return PolicyDecision(out, applied, "PASS", True, "eligible bounded context policy applied")

    # NPB 3-way: apply home-vs-away and draw residuals in log-probability space,
    # then softmax. There is no hard directional override.
    logits = np.log(np.clip(base, EPS, 1.0))
    applied: Dict[str, float] = {}
    for cfg in effect_rows:
        key = str(cfg.get("key", ""))
        obs = context.get(key) or {}
        if not _state_ok(obs):
            continue
        confidence = float(np.clip(obs.get("confidence", 1.0), 0.0, 1.0))
        cap = abs(float(cfg.get("cap", 0.0)))
        cls = cfg.get("class_index")
        target = str(cfg.get("target", "home")).lower()
        coef = float(cfg.get("coef", 0.0))
        shift = float(np.clip(coef * confidence, -cap, cap))
        if cls is not None:
            try:
                idx = int(cls)
            except Exception:
                idx = -1
            if not (0 <= idx < len(logits)):
                continue
            logits[idx] += shift
        elif target == "away" and len(logits) >= 2:
            logits[-1] += shift
        elif target == "draw" and len(logits) >= 3:
            logits[1] += shift
        else:
            logits[0] += shift
        if abs(shift) > 0:
            applied[key] = applied.get(key, 0.0) + shift
    logits -= np.max(logits)
    out = np.exp(np.clip(logits, -30.0, 30.0))
    out /= max(float(out.sum()), EPS)
    out = _prob_vector((1.0 - fusion_alpha) * base + fusion_alpha * out)
    return PolicyDecision(out, applied, "PASS", True, "eligible bounded context policy applied")


def self_test() -> Dict[str, Any]:
    base = np.array([0.55, 0.20, 0.25])
    ctx = {
        "starter": {"state": "VERIFIED", "confidence": 0.9},
        "lineup": {"state": "UNKNOWN", "confidence": 1.0},
    }
    policy = {
        "status": "PASS",
        "eligible": True,
        "effects": {
            "starter": {"coef": 0.08, "cap": 0.12, "target": "home"},
            "lineup": {"coef": 0.20, "cap": 0.20, "target": "away"},
        },
    }
    d = apply_matchday_policy(base, ctx, policy=policy)
    assert np.isclose(d.probabilities.sum(), 1.0)
    assert "starter" in d.applied_effects
    assert "lineup" not in d.applied_effects

    multi = apply_matchday_policy(
        base,
        {"ctx_STARTER_CHANGED": {"state": "VERIFIED", "confidence": 1.0}},
        policy={
            "status": "PASS",
            "eligible": True,
            "fusion_alpha": 1.0,
            "effects": [
                {"key": "ctx_STARTER_CHANGED", "coef": 0.05, "cap": 0.10, "class_index": 0},
                {"key": "ctx_STARTER_CHANGED", "coef": -0.03, "cap": 0.10, "class_index": 2},
            ],
        },
    )
    assert "ctx_STARTER_CHANGED" in multi.applied_effects
    assert np.isclose(multi.probabilities.sum(), 1.0)

    identity = apply_matchday_policy(base, {"starter": {"state": "UNKNOWN"}}, policy={
        "status": "PASS", "eligible": True, "effects": {"starter": {"coef": 0.30, "cap": 0.30}}
    })
    assert np.allclose(identity.probabilities, base)

    return {
        "status": "PASS",
        "applied_effects": d.applied_effects,
        "probabilities": d.probabilities.tolist(),
    }


if __name__ == "__main__":
    print(json.dumps(self_test(), ensure_ascii=False, sort_keys=True))
