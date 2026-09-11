"""Baseball Research Engine entry point for the v4.4 bridge.

This first integration stage is intentionally compatibility-only.  It routes
research calls through ``V44BaseballBacktest`` while keeping the legacy
``BaseballBacktest`` available as the reference implementation.  No candidate
is promoted here; Development OOS / Candidate Lock is a later stage.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from baseball_backtest import BaseballBacktest
from research.v44_bridge import V44BaseballBacktest


@dataclass(frozen=True)
class CompatibilityResult:
    """Result of a baseline-vs-bridge compatibility check."""

    passed: bool
    method: str
    details: dict[str, Any]


class BaseballResearchEngine:
    """Research entry point that opts into the v4.4 Baseball bridge.

    The engine deliberately does not replace the legacy implementation.  It
    creates a baseline instance and a bridge instance and can compare their
    outputs on the same inputs.  This makes the integration auditable before
    any research candidate is allowed to change production behavior.
    """

    def __init__(
        self,
        data_dir=None,
        baseline_factory: Callable[..., BaseballBacktest] = BaseballBacktest,
        bridge_factory: Callable[..., V44BaseballBacktest] = V44BaseballBacktest,
    ) -> None:
        kwargs = {} if data_dir is None else {"data_dir": data_dir}
        self.baseline = baseline_factory(**kwargs)
        self.bridge = bridge_factory(**kwargs)

    def fit_ensemble_compatibility(
        self, X: pd.DataFrame, y: pd.Series | np.ndarray, league: str
    ) -> CompatibilityResult:
        """Run the same fit through baseline and bridge and compare outputs."""
        X_base = X.copy(deep=True)
        X_bridge = X.copy(deep=True)
        y_base = np.asarray(y).copy()
        y_bridge = np.asarray(y).copy()

        base = self.baseline.fit_ensemble(X_base, y_base, league)
        bridged = self.bridge.fit_ensemble(X_bridge, y_bridge, league)

        base_models, base_scores, base_best = base
        bridge_models, bridge_scores, bridge_best = bridged

        score_equal = base_scores == bridge_scores
        best_equal = base_best == bridge_best
        model_names_equal = list(base_models.keys()) == list(bridge_models.keys())
        passed = bool(score_equal and best_equal and model_names_equal)

        return CompatibilityResult(
            passed=passed,
            method="fit_ensemble",
            details={
                "score_equal": score_equal,
                "best_model_equal": best_equal,
                "model_names_equal": model_names_equal,
                "baseline_best_model": base_best,
                "bridge_best_model": bridge_best,
                "baseline_validation_scores": base_scores,
                "bridge_validation_scores": bridge_scores,
                "bridge_audit_tail": self.bridge.audit[-1:] if self.bridge.audit else [],
            },
        )

    def run_walkforward_compatibility(
        self, games: pd.DataFrame, league: str
    ) -> CompatibilityResult:
        """Compare legacy and bridge walk-forward outputs on identical inputs."""
        base = self.baseline.run_walkforward(games.copy(deep=True), league)
        bridge = self.bridge.run_walkforward(games.copy(deep=True), league)

        if list(base.columns) != list(bridge.columns) or len(base) != len(bridge):
            return CompatibilityResult(
                passed=False,
                method="run_walkforward",
                details={
                    "shape_equal": base.shape == bridge.shape,
                    "columns_equal": list(base.columns) == list(bridge.columns),
                    "baseline_shape": base.shape,
                    "bridge_shape": bridge.shape,
                },
            )

        try:
            pd.testing.assert_frame_equal(
                base.reset_index(drop=True),
                bridge.reset_index(drop=True),
                check_dtype=True,
                check_exact=True,
            )
            passed = True
            diff = None
        except AssertionError as exc:
            passed = False
            diff = str(exc)

        return CompatibilityResult(
            passed=passed,
            method="run_walkforward",
            details={
                "shape_equal": True,
                "columns_equal": True,
                "exact_frame_equal": passed,
                "difference": diff,
                "bridge_audit_tail": self.bridge.audit[-1:] if self.bridge.audit else [],
            },
        )
