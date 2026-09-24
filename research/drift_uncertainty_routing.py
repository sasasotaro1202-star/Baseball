"""Research-only drift/uncertainty-aware expert routing and online recalibration.

Design goals:
- past-only / predict-then-update semantics;
- soft routing, never hard model switching;
- drift changes how much recent expert performance matters;
- high predictive disagreement pulls weights toward a conservative consensus;
- calibration is updated only after outcomes become available;
- production promotion remains external and gated by OOS/frozen-holdout validation.

No external services or non-standard dependencies are required.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


_EPS = 1e-12


@dataclass(frozen=True)
class RoutingConfig:
    """Conservative defaults for sequential probabilistic routing."""

    short_window: int = 24
    long_window: int = 96
    decay_half_life: float = 24.0
    drift_recent_mix: float = 0.85
    min_weight: float = 0.03
    inertia: float = 0.65
    drift_inertia_relax: float = 0.35
    uncertainty_uniform_mix: float = 0.70
    # Disagreement is a stronger proxy for epistemic/model uncertainty than
    # consensus entropy. Entropy is retained as an ambiguity proxy, but with
    # less influence on the routing fallback.
    uncertainty_disagreement_mix: float = 0.75
    uncertainty_entropy_mix: float = 0.25
    uncertainty_anchor_mix: float = 1.0
    drift_temperature: float = 0.30
    temperature_min: float = 0.65
    temperature_max: float = 1.90
    calibration_alpha: float = 0.15
    calibration_min_samples: int = 30
    calibration_validation_fraction: float = 0.30
    calibration_min_validation: int = 15
    calibration_min_improvement: float = 0.0005
    calibration_max_step: float = 0.15


@dataclass(frozen=True)
class RoutingResult:
    weights: np.ndarray
    short_loss: np.ndarray
    long_loss: np.ndarray
    drift_score: float
    disagreement: float
    predictive_entropy: float
    uncertainty: float


def _as_prob_matrix(probs: np.ndarray) -> np.ndarray:
    p = np.asarray(probs, dtype=float)
    if p.ndim != 2 or p.shape[0] == 0 or p.shape[1] < 2:
        raise ValueError("expert probabilities must have shape (n_experts, n_classes>=2)")
    if not np.all(np.isfinite(p)):
        raise ValueError("expert probabilities contain non-finite values")
    p = np.clip(p, _EPS, 1.0)
    p /= p.sum(axis=1, keepdims=True)
    return p


def _stable_softmax(scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    s = np.asarray(scores, dtype=float).reshape(-1)
    if not np.all(np.isfinite(s)):
        raise ValueError("routing scores contain non-finite values")
    t = max(float(temperature), 1e-6)
    z = (s - np.max(s)) / t
    e = np.exp(np.clip(z, -50.0, 50.0))
    w = e / max(float(e.sum()), _EPS)
    return w


def _recency_weights(n: int, half_life: float) -> np.ndarray:
    if n <= 0:
        return np.empty(0, dtype=float)
    age = np.arange(n - 1, -1, -1, dtype=float)
    h = max(float(half_life), 1.0)
    w = np.exp(-np.log(2.0) * age / h)
    return w / max(float(w.sum()), _EPS)


def _finite_weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    v = np.asarray(values, dtype=float)
    m = np.isfinite(v)
    if not np.any(m):
        return float("nan")
    ww = np.asarray(weights, dtype=float)[m]
    vv = v[m]
    return float(np.sum(vv * ww) / max(float(np.sum(ww)), _EPS))


def decayed_expert_losses(
    history_logloss: np.ndarray,
    short_window: int = 24,
    long_window: int = 96,
    half_life: float = 24.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return recent and long-run loss for each expert using only past rows.

    history_logloss is chronological with shape (n_past_games, n_experts).
    NaN is allowed for an unavailable expert and is ignored, never imputed as
    zero. The latest row is the most recent already-resolved outcome.
    """

    h = np.asarray(history_logloss, dtype=float)
    if h.ndim != 2 or h.shape[1] == 0:
        raise ValueError("history_logloss must be a 2-D matrix")
    if h.shape[0] == 0:
        k = h.shape[1]
        return np.full(k, np.nan), np.full(k, np.nan)

    recent = h[-max(1, int(short_window)) :]
    long = h[-max(1, min(int(long_window), len(h))) :]
    rw = _recency_weights(len(recent), half_life)
    lw = _recency_weights(len(long), half_life)

    short = np.array([_finite_weighted_mean(recent[:, j], rw) for j in range(h.shape[1])])
    long_loss = np.array([_finite_weighted_mean(long[:, j], lw) for j in range(h.shape[1])])

    # Fall back to the other horizon only when an expert has no observations.
    short = np.where(np.isfinite(short), short, long_loss)
    long_loss = np.where(np.isfinite(long_loss), long_loss, short)
    return short, long_loss


def feature_drift_score(
    reference: np.ndarray,
    current: np.ndarray,
) -> float:
    """Bounded covariate-drift score from pre-prediction numeric windows.

    This is deliberately target-free. It compares location and scale changes
    between a reference window and the currently available feature window.
    The result is mapped to [0, 1], with higher values indicating stronger
    distributional change. Missing/non-numeric columns should be removed before
    calling this function.
    """

    a = np.asarray(reference, dtype=float)
    b = np.asarray(current, dtype=float)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1] or a.shape[1] == 0:
        raise ValueError("reference/current must be 2-D arrays with matching feature count")
    if len(a) < 2 or len(b) < 2:
        return 0.0
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("drift windows contain non-finite values")

    ma = a.mean(axis=0)
    mb = b.mean(axis=0)
    sa = a.std(axis=0, ddof=1)
    sb = b.std(axis=0, ddof=1)
    scale = np.maximum(sa, 1e-6)
    location_shift = np.abs(mb - ma) / scale
    scale_shift = np.abs(np.log((sb + 1e-6) / (sa + 1e-6)))
    raw = float(np.nanmean(0.5 * location_shift + 0.5 * scale_shift))
    return float(np.clip(1.0 - np.exp(-raw / 2.0), 0.0, 1.0))


def ensemble_uncertainty(expert_probs: np.ndarray) -> Tuple[float, float, float]:
    """Return (disagreement, normalized predictive entropy, combined uncertainty)."""

    p = _as_prob_matrix(expert_probs)
    disagreement = float(np.mean(np.std(p, axis=0)))
    mean_p = np.mean(p, axis=0)
    entropy = float(-np.sum(mean_p * np.log(np.clip(mean_p, _EPS, 1.0))))
    max_entropy = float(np.log(p.shape[1]))
    normalized_entropy = entropy / max(max_entropy, _EPS)
    # Treat these as proxies, not a formal identifiability result:
    # disagreement ≈ epistemic/model uncertainty, consensus entropy ≈
    # outcome ambiguity (aleatoric-like uncertainty). The routing fallback
    # therefore reacts primarily to disagreement rather than blindly flattening
    # every intrinsically difficult game.
    disagreement_norm = float(np.clip(disagreement / 0.25, 0.0, 1.0))
    uncertainty = float(np.clip(
        0.75 * disagreement_norm + 0.25 * normalized_entropy,
        0.0, 1.0
    ))
    return disagreement, normalized_entropy, uncertainty


def route_experts(
    expert_probs: np.ndarray,
    history_logloss: np.ndarray,
    drift_score: float = 0.0,
    previous_weights: Optional[np.ndarray] = None,
    config: RoutingConfig = RoutingConfig(),
) -> RoutingResult:
    """Softly route experts using recent-vs-long performance and uncertainty.

    High drift increases the influence of recent performance and relaxes
    inertia, but never performs a hard switch. High disagreement/entropy moves
    the result toward a conservative consensus to avoid overconfident routing.
    """

    p = _as_prob_matrix(expert_probs)
    k = p.shape[0]
    hist = np.asarray(history_logloss, dtype=float)
    if hist.ndim != 2 or hist.shape[1] != k:
        raise ValueError("history_logloss expert count must match expert_probs")
    d = float(np.clip(drift_score, 0.0, 1.0))

    short, long_loss = decayed_expert_losses(
        hist,
        short_window=config.short_window,
        long_window=config.long_window,
        half_life=config.decay_half_life,
    )

    finite = np.isfinite(short) & np.isfinite(long_loss)
    if not np.any(finite):
        base = np.full(k, 1.0 / k)
    else:
        recency_mix = float(np.clip(config.drift_recent_mix * d, 0.0, 1.0))
        blended = np.where(
            finite,
            (1.0 - recency_mix) * long_loss + recency_mix * short,
            np.nan,
        )
        fill = float(np.nanmean(blended[finite]))
        blended = np.where(np.isfinite(blended), blended, fill)
        # Smaller loss -> larger score. Temperature becomes modestly sharper
        # under strong drift, but remains bounded.
        temp = max(config.temperature_min, 1.0 - config.drift_temperature * d)
        base = _stable_softmax(-blended, temperature=temp)

    disagreement, predictive_entropy, uncertainty = ensemble_uncertainty(p)
    disagreement_norm = float(np.clip(disagreement / 0.25, 0.0, 1.0))
    ambiguity_norm = float(np.clip(predictive_entropy, 0.0, 1.0))
    uncertainty_for_fallback = float(np.clip(
        config.uncertainty_disagreement_mix * disagreement_norm
        + config.uncertainty_entropy_mix * ambiguity_norm,
        0.0, 1.0
    ))
    uniform_mix = float(np.clip(
        config.uncertainty_uniform_mix * uncertainty_for_fallback,
        0.0, 0.80
    ))
    if previous_weights is not None:
        prev = np.asarray(previous_weights, dtype=float).reshape(-1)
        if len(prev) != k or not np.all(np.isfinite(prev)) or np.any(prev < 0):
            raise ValueError("previous_weights is invalid")
        prev = prev / max(float(prev.sum()), _EPS)
        # Under high epistemic uncertainty, shrink toward the previously
        # validated routing state rather than an arbitrary uniform prior.
        # This preserves the stable incumbent while still allowing new experts
        # to contribute. At startup, uniform remains the neutral anchor.
        anchor = float(np.clip(config.uncertainty_anchor_mix, 0.0, 1.0)) * prev
        anchor += (1.0 - float(np.clip(config.uncertainty_anchor_mix, 0.0, 1.0))) * (1.0 / k)
    else:
        anchor = np.full(k, 1.0 / k)
    weights = (1.0 - uniform_mix) * base + uniform_mix * anchor

    if previous_weights is not None:
        inertia = np.clip(config.inertia - config.drift_inertia_relax * d, 0.0, 0.95)
        weights = inertia * prev + (1.0 - inertia) * weights

    floor = min(float(config.min_weight), 0.98 / k)
    weights = np.maximum(weights, floor)
    weights /= max(float(weights.sum()), _EPS)

    return RoutingResult(
        weights=weights,
        short_loss=short,
        long_loss=long_loss,
        drift_score=d,
        disagreement=disagreement,
        predictive_entropy=predictive_entropy,
        uncertainty=uncertainty,
    )


def mix_expert_probabilities(expert_probs: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Return one normalized ensemble probability vector."""

    p = _as_prob_matrix(expert_probs)
    w = np.asarray(weights, dtype=float).reshape(-1)
    if len(w) != p.shape[0] or not np.all(np.isfinite(w)) or np.any(w < 0):
        raise ValueError("weights are invalid")
    w = w / max(float(w.sum()), _EPS)
    out = np.sum(p * w[:, None], axis=0)
    out = np.clip(out, _EPS, 1.0)
    return out / max(float(out.sum()), _EPS)


def apply_temperature(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    """Apply multiclass temperature to already normalized probabilities."""

    p = np.asarray(probabilities, dtype=float).reshape(-1)
    if len(p) < 2 or not np.all(np.isfinite(p)) or np.any(p < 0):
        raise ValueError("probabilities are invalid")
    p = np.clip(p, _EPS, 1.0)
    p /= max(float(p.sum()), _EPS)
    t = float(np.clip(temperature, 0.1, 10.0))
    q = np.power(p, 1.0 / t)
    return q / max(float(q.sum()), _EPS)


def _logloss(probabilities: np.ndarray, y: np.ndarray) -> float:
    p = np.asarray(probabilities, dtype=float)
    labels = np.asarray(y, dtype=int).reshape(-1)
    if p.ndim != 2 or len(labels) != len(p):
        raise ValueError("probabilities/labels length mismatch")
    if np.any(labels < 0) or np.any(labels >= p.shape[1]):
        raise ValueError("labels are outside probability columns")
    return float(np.mean(-np.log(np.clip(p[np.arange(len(labels)), labels], _EPS, 1.0))))


def fit_temperature(
    probabilities: np.ndarray,
    y: np.ndarray,
    min_samples: int = 30,
    t_min: float = 0.65,
    t_max: float = 1.90,
) -> float:
    """Fit temperature using already-resolved past outcomes only."""

    p = np.asarray(probabilities, dtype=float)
    labels = np.asarray(y, dtype=int).reshape(-1)
    if p.ndim != 2 or len(labels) != len(p) or len(labels) < int(min_samples):
        return 1.0
    if not np.all(np.isfinite(p)):
        raise ValueError("calibration probabilities contain non-finite values")
    best_t = 1.0
    best = float("inf")
    # Dense enough for a lightweight research/control path; no hidden optimizer.
    for t in np.linspace(float(t_min), float(t_max), 51):
        q = np.array([apply_temperature(row, float(t)) for row in p])
        loss = _logloss(q, labels) + 0.01 * (float(t) - 1.0) ** 2
        if loss < best:
            best = loss
            best_t = float(t)
    return best_t


class AdaptiveTemperatureCalibrator:
    """Predict-then-update temperature tracker.

    predict() never consumes the current outcome. update() may be called only
    after that forecast's outcome is actually known.
    """

    def __init__(
        self,
        temperature: float = 1.0,
        config: RoutingConfig = RoutingConfig(),
    ):
        self.temperature = float(np.clip(temperature, config.temperature_min, config.temperature_max))
        self.config = config
        self.accepted_updates = 0
        self.rejected_updates = 0

    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        return apply_temperature(probabilities, self.temperature)

    def update(self, past_probabilities: np.ndarray, past_outcomes: np.ndarray) -> float:
        """Update from raw pre-calibration probabilities, then expose new temperature.

        The input probabilities must be the mixed expert probabilities BEFORE the
        current temperature is applied. A chronological fit/validation split is
        used so a candidate temperature only moves when it improves log loss on
        later resolved rows. This avoids recursively fitting a temperature on
        probabilities that were already temperature-scaled.
        """
        p = np.asarray(past_probabilities, dtype=float)
        y = np.asarray(past_outcomes, dtype=int).reshape(-1)
        min_total = max(
            self.config.calibration_min_samples + self.config.calibration_min_validation,
            self.config.calibration_min_samples * 2,
        )
        if len(y) < min_total:
            return self.temperature
        if p.ndim != 2 or len(p) != len(y) or not np.all(np.isfinite(p)):
            raise ValueError("calibration probabilities/outcomes are invalid")

        frac = float(np.clip(self.config.calibration_validation_fraction, 0.15, 0.50))
        val_n = max(self.config.calibration_min_validation, int(round(len(y) * frac)))
        if val_n >= len(y):
            return self.temperature
        fit_n = len(y) - val_n
        fit_p, fit_y = p[:fit_n], y[:fit_n]
        val_p, val_y = p[fit_n:], y[fit_n:]
        candidate = fit_temperature(
            fit_p,
            fit_y,
            min_samples=self.config.calibration_min_samples,
            t_min=self.config.temperature_min,
            t_max=self.config.temperature_max,
        )
        current_loss = _logloss(
            np.array([apply_temperature(row, self.temperature) for row in val_p]),
            val_y,
        )
        candidate_loss = _logloss(
            np.array([apply_temperature(row, candidate) for row in val_p]),
            val_y,
        )
        if candidate_loss + self.config.calibration_min_improvement >= current_loss:
            self.rejected_updates += 1
            return self.temperature

        delta = float(np.clip(
            candidate - self.temperature,
            -self.config.calibration_max_step,
            self.config.calibration_max_step,
        ))
        self.temperature = float(np.clip(
            self.temperature + self.config.calibration_alpha * delta,
            self.config.temperature_min,
            self.config.temperature_max,
        ))
        self.accepted_updates += 1
        return self.temperature


def route_and_recalibrate(
    expert_probs: np.ndarray,
    history_logloss: np.ndarray,
    drift_score: float = 0.0,
    previous_weights: Optional[np.ndarray] = None,
    calibrator: Optional[AdaptiveTemperatureCalibrator] = None,
    config: RoutingConfig = RoutingConfig(),
) -> Tuple[np.ndarray, RoutingResult, float]:
    """Research helper: route -> mix -> optional existing temperature.

    This function is prediction-time only. Outcome-dependent calibration updates
    must happen later through AdaptiveTemperatureCalibrator.update().
    """

    result = route_experts(
        expert_probs=expert_probs,
        history_logloss=history_logloss,
        drift_score=drift_score,
        previous_weights=previous_weights,
        config=config,
    )
    mixed = mix_expert_probabilities(expert_probs, result.weights)
    t = calibrator.temperature if calibrator is not None else 1.0
    return apply_temperature(mixed, t), result, t


def self_test() -> dict:
    """Deterministic no-network smoke test used by CI."""

    rng = np.random.default_rng(7)
    hist = np.zeros((120, 3), dtype=float)
    # A is long-run stable, B becomes better recently under drift, C is noisy.
    hist[:, 0] = 0.60 + 0.02 * rng.normal(size=120)
    hist[:, 1] = 0.62 + 0.02 * rng.normal(size=120)
    hist[-24:, 1] -= 0.16
    hist[:, 2] = 0.68 + 0.03 * rng.normal(size=120)

    probs = np.array([
        [0.58, 0.42],
        [0.75, 0.25],
        [0.55, 0.45],
    ])
    low = route_experts(probs, hist, drift_score=0.0)
    high = route_experts(probs, hist, drift_score=0.9)

    assert np.isclose(low.weights.sum(), 1.0)
    assert np.isclose(high.weights.sum(), 1.0)
    assert high.weights[1] >= low.weights[1]
    assert high.uncertainty >= 0.0

    y = np.array([0, 1] * 30)
    overconf = np.tile(np.array([[0.90, 0.10], [0.10, 0.90]]), (30, 1))
    t = fit_temperature(overconf, y, min_samples=30)
    q = apply_temperature(overconf[0], t)
    assert 0.65 <= t <= 1.90
    assert np.isclose(q.sum(), 1.0)

    c = AdaptiveTemperatureCalibrator()
    before = c.temperature
    after = c.update(overconf, y)
    assert np.isfinite(after)
    assert abs(after - before) <= 0.15 + 1e-9
    assert c.accepted_updates + c.rejected_updates <= 1

    return {
        "status": "PASS",
        "high_drift_weight_delta_for_recent_specialist": float(high.weights[1] - low.weights[1]),
        "temperature": float(t),
        "adaptive_temperature": float(after),
    }


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), ensure_ascii=False, sort_keys=True))
