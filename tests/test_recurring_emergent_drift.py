import numpy as np

from research.recurring_emergent_drift import classify_drift, guard_routing_weights


def _window(mean, scale=1.0, n=12):
    rng = np.random.default_rng(7)
    return rng.normal(loc=mean, scale=scale, size=(n, 3))


def test_recurring_state_is_detected_from_older_window_only():
    old = _window([0.0, 1.0, 2.0])
    cur = _window([0.0, 1.0, 2.0])
    result = classify_drift([old], cur)
    assert result.drift_type == "RECURRING"
    assert result.recurrence_score > 0.65
    assert result.novelty_score < 0.35


def test_novel_state_is_emergent():
    old = _window([0.0, 1.0, 2.0])
    cur = _window([8.0, 9.0, 10.0])
    result = classify_drift([old], cur)
    assert result.drift_type == "EMERGENT"
    assert result.novelty_score >= 0.70


def test_no_history_fails_closed_to_unknown():
    result = classify_drift([], _window([0.0, 1.0, 2.0]))
    assert result.drift_type == "UNKNOWN"
    assert result.recurrence_score == 0.0
    assert result.novelty_score == 1.0


def test_emergent_high_uncertainty_limits_weight_movement():
    prev = np.array([0.70, 0.20, 0.10])
    proposed = np.array([0.10, 0.20, 0.70])
    out = guard_routing_weights(
        proposed,
        prev,
        recurrence_score=0.05,
        novelty_score=0.95,
        uncertainty=0.95,
        max_change_emergent=0.12,
    )
    assert np.isclose(out.sum(), 1.0)
    assert np.max(np.abs(out - prev)) <= 0.12 + 1e-9


def test_recurring_state_allows_more_adaptation_than_emergent():
    prev = np.array([0.70, 0.20, 0.10])
    proposed = np.array([0.10, 0.20, 0.70])
    recurring = guard_routing_weights(
        proposed, prev, recurrence_score=0.95, novelty_score=0.05, uncertainty=0.20
    )
    emergent = guard_routing_weights(
        proposed, prev, recurrence_score=0.05, novelty_score=0.95, uncertainty=0.95
    )
    assert np.linalg.norm(recurring - prev) > np.linalg.norm(emergent - prev)
