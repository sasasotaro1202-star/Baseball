#!/usr/bin/env python3
"""Research-only local-competence routing in prediction space.

The current query's expert probability vector defines a prediction-space
region-of-competence. Only already-resolved historical rows are eligible as
neighbors, so the procedure is predict-then-update and PIT-safe by design.

This is a candidate layer: it never mutates production state.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


_EPS = 1e-12


@dataclass(frozen=True)
class LocalCompetenceConfig:
    k_neighbors: int = 32
    temperature: float = 0.12
    distance_power: float = 1.0
    prior_mix: float = 0.35
    min_history: int = 40
    min_local_observations: int = 8
    competence_floor: float = 0.05
    recency_half_life: float = 48.0


@dataclass(frozen=True)
class LocalCompetenceResult:
    weights: np.ndarray
    neighbor_count: int
    mean_distance: float
    local_loss: np.ndarray


def _normalize_rows(p: np.ndarray) -> np.ndarray:
    x = np.asarray(p, dtype=float)
    if x.ndim != 2 or x.shape[1] < 2 or not np.all(np.isfinite(x)):
        raise ValueError("probability matrix must be finite 2-D with >=2 classes")
    x = np.clip(x, _EPS, 1.0)
    return x / np.maximum(x.sum(axis=1, keepdims=True), _EPS)


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    mask = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not np.any(mask):
        return float("nan")
    return float(np.sum(v[mask] * w[mask]) / np.sum(w[mask]))


def local_competence_weights(
    current_expert_probs: np.ndarray,
    historical_expert_probs: np.ndarray,
    historical_outcomes: np.ndarray,
    *,
    history_expert_logloss: Optional[np.ndarray] = None,
    config: LocalCompetenceConfig = LocalCompetenceConfig(),
) -> LocalCompetenceResult:
    """Estimate per-expert competence from prediction-space neighbors.

    Args:
        current_expert_probs: shape (n_experts, n_classes), current pre-outcome
            probabilities.
        historical_expert_probs: shape (n_past, n_experts, n_classes), ordered
            chronologically and containing only already-resolved rows.
        historical_outcomes: shape (n_past,), resolved labels.
        history_expert_logloss: optional shape (n_past, n_experts), already
            scored historical losses. If omitted, they are computed directly.

    Returns:
        Soft weights that can be mixed with a global routing prior.
    """
    cur = _normalize_rows(current_expert_probs)
    hist = np.asarray(historical_expert_probs, dtype=float)
    y = np.asarray(historical_outcomes, dtype=int).reshape(-1)

    if hist.ndim != 3:
        raise ValueError("historical_expert_probs must be 3-D")
    if hist.shape[1:] != cur.shape:
        raise ValueError("historical expert/class dimensions do not match current")
    if len(hist) != len(y):
        raise ValueError("historical probabilities/outcomes length mismatch")
    if len(y) and (np.any(y < 0) or np.any(y >= cur.shape[1])):
        raise ValueError("historical outcomes outside class range")

    n_past, n_experts, _ = hist.shape
    if n_past < int(config.min_history):
        return LocalCompetenceResult(
            weights=np.full(n_experts, 1.0 / n_experts),
            neighbor_count=0,
            mean_distance=float("nan"),
            local_loss=np.full(n_experts, np.nan),
        )

    hist = np.stack([_normalize_rows(hist[i]) for i in range(n_past)], axis=0)

    # Prediction-space distance: compare the full probability vector produced by
    # all experts for each past case. This is target-free at the current row.
    diff = hist - cur[None, :, :]
    d = np.sqrt(np.mean(diff * diff, axis=(1, 2)))
    k = min(max(1, int(config.k_neighbors)), n_past)
    idx = np.argpartition(d, k - 1)[:k]
    idx = idx[np.argsort(d[idx], kind="mergesort")]
    distances = d[idx]

    # Closest cases receive exponentially decayed weight, with additional
    # chronology decay so a very old regime cannot dominate a recent analogue.
    temp = max(float(config.temperature), 1e-4)
    sim = np.exp(-distances / temp)
    age = np.arange(n_past - 1, n_past - len(idx) - 1, -1, dtype=float)
    # Recover chronological age from the selected absolute indices.
    age = (n_past - 1) - idx.astype(float)
    half_life = max(float(config.recency_half_life), 1.0)
    recency = np.exp(-np.log(2.0) * age / half_life)
    nw = sim * recency
    nw = nw / max(float(nw.sum()), _EPS)

    if history_expert_logloss is None:
        losses = -np.log(
            np.clip(hist[idx, np.arange(n_experts)[None, :], y[idx, None]], _EPS, 1.0)
        )
    else:
        hl = np.asarray(history_expert_logloss, dtype=float)
        if hl.shape != (n_past, n_experts) or not np.all(np.isfinite(hl)):
            raise ValueError("history_expert_logloss shape/values are invalid")
        losses = hl[idx]

    local_loss = np.asarray([
        _weighted_mean(losses[:, j], nw)
        for j in range(n_experts)
    ], dtype=float)

    finite = np.isfinite(local_loss)
    if not np.any(finite):
        return LocalCompetenceResult(
            weights=np.full(n_experts, 1.0 / n_experts),
            neighbor_count=int(len(idx)),
            mean_distance=float(np.mean(distances)),
            local_loss=local_loss,
        )

    # Convert lower local loss -> higher competence, with a conservative floor.
    anchor = float(np.clip(config.prior_mix, 0.0, 1.0))
    scores = np.zeros(n_experts, dtype=float)
    fill = float(np.nanmean(local_loss[finite]))
    safe_loss = np.where(finite, local_loss, fill)
    scaled = -(safe_loss - np.min(safe_loss))
    scores = np.exp(np.clip(scaled / max(float(config.distance_power), _EPS), -30.0, 30.0))
    scores = scores / max(float(scores.sum()), _EPS)
    floor = min(float(config.competence_floor), 0.95 / n_experts)
    scores = np.maximum(scores, floor)
    scores = scores / max(float(scores.sum()), _EPS)

    prior = np.full(n_experts, 1.0 / n_experts)
    weights = anchor * prior + (1.0 - anchor) * scores
    weights = np.maximum(weights, floor)
    weights = weights / max(float(weights.sum()), _EPS)

    return LocalCompetenceResult(
        weights=weights,
        neighbor_count=int(len(idx)),
        mean_distance=float(np.mean(distances)),
        local_loss=local_loss,
    )


def mix_with_global_prior(
    local_weights: np.ndarray,
    global_weights: np.ndarray,
    local_observation_count: int,
    *,
    min_local_observations: int = 8,
    local_mix_max: float = 0.65,
) -> np.ndarray:
    """Blend local competence with a pre-existing global routing prior."""
    lw = np.asarray(local_weights, dtype=float).reshape(-1)
    gw = np.asarray(global_weights, dtype=float).reshape(-1)
    if len(lw) != len(gw) or len(lw) == 0:
        raise ValueError("local/global weight lengths differ")
    if (
        not np.all(np.isfinite(lw))
        or not np.all(np.isfinite(gw))
        or np.any(lw < 0)
        or np.any(gw < 0)
        or int(local_observation_count) < int(min_local_observations)
    ):
        return gw / max(float(gw.sum()), _EPS)
    lw = lw / max(float(lw.sum()), _EPS)
    gw = gw / max(float(gw.sum()), _EPS)
    coverage = np.clip(
        (float(local_observation_count) - min_local_observations) / 24.0,
        0.0,
        1.0,
    )
    mix = float(np.clip(local_mix_max, 0.0, 0.80)) * coverage
    out = (1.0 - mix) * gw + mix * lw
    return out / max(float(out.sum()), _EPS)


def self_test() -> dict:
    # Expert 0 is better on probability-space cases near the current query;
    # expert 1 is globally better in the distant cases.
    current = np.array([
        [0.82, 0.18],
        [0.60, 0.40],
        [0.55, 0.45],
    ])
    rows = []
    y = []
    rng = np.random.default_rng(11)
    for i in range(60):
        base = np.array([0.80, 0.20]) + rng.normal(0, 0.015, 2)
        base = np.clip(base, 0.05, 0.95); base /= base.sum()
        rows.append(np.stack([base, np.array([0.55, 0.45]), np.array([0.50, 0.50])]))
        y.append(0)
    hist = np.asarray(rows)
    y = np.asarray(y)
    ll = np.zeros((60, 3))
    ll[:, 0] = 0.10
    ll[:, 1] = 0.55
    ll[:, 2] = 0.75
    result = local_competence_weights(
        current,
        hist,
        y,
        history_expert_logloss=ll,
        config=LocalCompetenceConfig(min_history=20, k_neighbors=24),
    )
    assert result.neighbor_count == 24
    assert np.isclose(result.weights.sum(), 1.0)
    assert result.weights[0] > result.weights[1] > result.weights[2]

    mixed = mix_with_global_prior(
        result.weights,
        np.array([0.20, 0.60, 0.20]),
        result.neighbor_count,
    )
    assert np.isclose(mixed.sum(), 1.0)
    assert mixed[0] > 0.20

    return {
        "status": "PASS",
        "neighbor_count": result.neighbor_count,
        "mean_distance": result.mean_distance,
        "local_weights": result.weights.tolist(),
        "mixed_weights": mixed.tolist(),
    }


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), ensure_ascii=False, sort_keys=True))
