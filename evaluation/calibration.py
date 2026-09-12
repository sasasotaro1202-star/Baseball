"""Calibration utilities used at the final probability boundary.

The calibrator is deliberately independent from model training so a frozen
calibration artifact can be versioned and applied to production predictions.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TemperatureCalibration:
    temperature: float = 1.0
    version: str = "temperature-v1"

    def transform(self, probabilities: Any) -> np.ndarray:
        p = np.asarray(probabilities, dtype=float)
        if p.ndim != 2:
            raise ValueError("probabilities must be 2D")
        if np.any(p < 0) or np.any(p > 1):
            raise ValueError("probabilities must be in [0,1]")
        if self.temperature <= 0 or not math.isfinite(self.temperature):
            raise ValueError("temperature must be positive and finite")
        logits = np.log(np.clip(p, 1e-15, 1.0)) / self.temperature
        logits -= logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        return exp / exp.sum(axis=1, keepdims=True)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def fit_temperature(probabilities: Any, y_true: Any, *, grid: np.ndarray | None = None) -> TemperatureCalibration:
    """Fit temperature on a development/calibration set only."""
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(y_true, dtype=int)
    if p.ndim != 2 or len(p) != len(y):
        raise ValueError("probabilities and y_true are incompatible")
    if np.any(y < 0) or np.any(y >= p.shape[1]):
        raise ValueError("y_true contains an invalid class")
    candidates = grid if grid is not None else np.geomspace(0.5, 3.0, 61)
    best_t, best_loss = 1.0, float("inf")
    for t in candidates:
        if t <= 0:
            continue
        q = TemperatureCalibration(float(t)).transform(p)
        loss = float(-np.mean(np.log(np.clip(q[np.arange(len(y)), y], 1e-15, 1.0))))
        if loss < best_loss:
            best_t, best_loss = float(t), loss
    return TemperatureCalibration(best_t)


def calibration_report(y_true: Any, raw: Any, calibrated: Any) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    r = np.asarray(raw, dtype=float)
    c = np.asarray(calibrated, dtype=float)
    if len(y) != len(r) or len(y) != len(c):
        raise ValueError("calibration arrays have incompatible lengths")
    raw_nll = float(-np.mean(np.log(np.clip(r[np.arange(len(y)), y], 1e-15, 1.0))))
    cal_nll = float(-np.mean(np.log(np.clip(c[np.arange(len(y)), y], 1e-15, 1.0))))
    return {"raw_logloss": raw_nll, "calibrated_logloss": cal_nll, "improvement": raw_nll - cal_nll, "rows": int(len(y))}
