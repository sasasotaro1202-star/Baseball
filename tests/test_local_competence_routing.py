from pathlib import Path

import numpy as np

from research.local_competence_routing import (
    LocalCompetenceConfig,
    local_competence_weights,
    mix_with_global_prior,
)


def test_local_competence_is_prediction_then_update_safe():
    current = np.array([[0.80, 0.20], [0.55, 0.45]])
    history = np.array([
        [[0.81, 0.19], [0.60, 0.40]],
        [[0.79, 0.21], [0.58, 0.42]],
        [[0.20, 0.80], [0.55, 0.45]],
    ])
    outcomes = np.array([0, 1, 1])
    result = local_competence_weights(
        current,
        history,
        outcomes,
        config=LocalCompetenceConfig(min_history=2, k_neighbors=2),
    )
    assert np.isclose(result.weights.sum(), 1.0)
    assert result.neighbor_count == 2


def test_global_fallback_when_local_history_is_insufficient():
    local = np.array([0.7, 0.3])
    global_prior = np.array([0.25, 0.75])
    mixed = mix_with_global_prior(
        local,
        global_prior,
        local_observation_count=3,
        min_local_observations=8,
    )
    assert np.allclose(mixed, global_prior / global_prior.sum())


def test_module_has_no_production_mutation_path():
    text = Path("research/local_competence_routing.py").read_text(encoding="utf-8")
    assert "production" in text.lower()
    assert "self_test" in text
