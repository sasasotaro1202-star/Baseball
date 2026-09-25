#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Research-only per-case pregame risk / uncertainty diagnostics.

This module does not change the production forecast. It turns an already
generated probabilistic forecast plus independently available evidence flags
into auditable case-level diagnostics:

  - predictive uncertainty (normalized entropy)
  - expert disagreement (normalized Jensen-Shannon divergence)
  - evidence/support gap
  - training-support/domain gap
  - bounded composite risk score
  - reversal-risk proxy

The composite is diagnostic only. Thresholds are intentionally simple and are
not an adoption/promotion rule; OOS calibration is required before any use for
routing, abstention, or probability adjustment.
"""
from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


EPS = 1e-12


def _clip_simplex(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        raise ValueError("probability vector must be finite and non-empty")
    arr = np.maximum(arr, EPS)
    total = float(arr.sum())
    if total <= 0.0:
        raise ValueError("probability vector has non-positive mass")
    return arr / total


def normalized_entropy(probabilities: Sequence[float]) -> float:
    """Entropy normalized to [0, 1] for k-class probabilities."""
    p = _clip_simplex(probabilities)
    if p.size <= 1:
        return 0.0
    h = -float(np.sum(p * np.log(p)))
    return float(np.clip(h / math.log(p.size), 0.0, 1.0))


def normalized_margin(probabilities: Sequence[float]) -> float:
    """1 - top-two margin, scaled so 1 means maximally ambiguous."""
    p = _clip_simplex(probabilities)
    if p.size == 1:
        return 0.0
    top2 = np.partition(p, -2)[-2:]
    margin = float(top2[-1] - top2[-2])
    return float(np.clip(1.0 - margin, 0.0, 1.0))


def normalized_js_divergence(distributions: Sequence[Sequence[float]]) -> float:
    """Normalized Jensen-Shannon divergence in [0, 1]."""
    if len(distributions) < 2:
        return 0.0
    matrix = np.vstack([_clip_simplex(x) for x in distributions])
    if len({row.size for row in matrix}) != 1:
        raise ValueError("all probability vectors must have the same width")
    mean = np.mean(matrix, axis=0)
    kl_terms = []
    for row in matrix:
        kl_terms.append(float(np.sum(row * np.log(row / np.maximum(mean, EPS)))))
    js = float(np.mean(kl_terms))
    max_js = math.log(2.0)
    return float(np.clip(js / max_js, 0.0, 1.0))


def _truthy(v: Any) -> bool:
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "yes", "y", "confirmed", "verified", "projected"}
    return bool(v)


def evidence_support(row: Mapping[str, Any]) -> tuple[float, int]:
    """Return support fraction and count from explicit pregame evidence flags."""
    checks: list[bool] = []

    confirmed = row.get("confirmed_starters", None)
    if confirmed is not None and not (isinstance(confirmed, float) and np.isnan(confirmed)):
        checks.append(_truthy(confirmed))

    starter_state = str(row.get("starter_confirmation_state", "") or "").strip().upper()
    if starter_state:
        checks.append(starter_state in {"CONFIRMED", "VERIFIED"})

    weather = row.get("weather_pit_safe", None)
    if weather is not None and not (isinstance(weather, float) and np.isnan(weather)):
        checks.append(_truthy(weather))

    lineup_present = row.get("ctx_lineup_present", None)
    if lineup_present is not None and not (isinstance(lineup_present, float) and np.isnan(lineup_present)):
        checks.append(_truthy(lineup_present))

    if not checks:
        return 0.0, 0
    return float(np.mean(checks)), len(checks)


def _extract_expert_distributions(row: Mapping[str, Any]) -> list[np.ndarray]:
    """Discover paired expert_*_{home,away[,draw]} columns from an OOS row."""
    keys = {str(k) for k in row.keys()}
    out: list[np.ndarray] = []
    for home_key in sorted(k for k in keys if k.startswith("expert_") and k.endswith("_home")):
        prefix = home_key[:-5]
        away_key = prefix + "_away"
        draw_key = prefix + "_draw"
        home = row.get(home_key, np.nan)
        away = row.get(away_key, np.nan)
        try:
            if pd.isna(home):
                continue
            home = float(home)
            if draw_key in keys and pd.notna(row.get(draw_key)):
                draw = float(row.get(draw_key))
                if away_key not in keys or pd.isna(row.get(away_key)):
                    continue
                away = float(away)
                out.append(_clip_simplex([home, draw, away]))
            else:
                if away_key in keys and pd.notna(row.get(away_key)):
                    away = float(away)
                else:
                    away = 1.0 - home
                out.append(_clip_simplex([home, away]))
        except (TypeError, ValueError):
            continue
    return out


def _forecast_distribution(row: Mapping[str, Any]) -> np.ndarray:
    vals: list[float]
    if pd.notna(row.get("pred_draw", np.nan)):
        vals = [float(row.get("pred_home")), float(row.get("pred_draw")), float(row.get("pred_away"))]
    else:
        vals = [float(row.get("pred_home")), float(row.get("pred_away"))]
    return _clip_simplex(vals)


def score_case(row: Mapping[str, Any]) -> dict[str, Any]:
    """Score one forecast row with fail-closed behavior for missing probabilities."""
    try:
        p = _forecast_distribution(row)
    except (TypeError, ValueError):
        return {
            "risk_available": False,
            "uncertainty": np.nan,
            "margin_risk": np.nan,
            "expert_disagreement": np.nan,
            "expert_count": 0,
            "evidence_support": 0.0,
            "evidence_count": 0,
            "training_support_gap": 1.0,
            "risk_score": np.nan,
            "reversal_risk_proxy": np.nan,
        }

    uncertainty = normalized_entropy(p)
    margin_risk = normalized_margin(p)
    experts = _extract_expert_distributions(row)
    disagreement = normalized_js_divergence(experts) if len(experts) >= 2 else 0.0
    support, support_count = evidence_support(row)

    training_included = row.get("training_included", True)
    training_gap = 0.0 if _truthy(training_included) else 1.0

    # Diagnostic weights only. They must be OOS-calibrated before operational use.
    evidence_gap = 1.0 - support if support_count else 1.0
    risk = (
        0.45 * uncertainty
        + 0.30 * disagreement
        + 0.15 * evidence_gap
        + 0.10 * training_gap
    )
    reversal = (
        0.60 * margin_risk
        + 0.25 * disagreement
        + 0.15 * evidence_gap
    )
    return {
        "risk_available": True,
        "uncertainty": float(uncertainty),
        "margin_risk": float(margin_risk),
        "expert_disagreement": float(disagreement),
        "expert_count": int(len(experts)),
        "evidence_support": float(support),
        "evidence_count": int(support_count),
        "training_support_gap": float(training_gap),
        "risk_score": float(np.clip(risk, 0.0, 1.0)),
        "reversal_risk_proxy": float(np.clip(reversal, 0.0, 1.0)),
    }


def score_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Add risk diagnostics without modifying forecast probabilities."""
    if df.empty:
        return df.copy()
    rows = [score_case(row) for row in df.to_dict("records")]
    scored = pd.DataFrame(rows, index=df.index)
    out = df.copy()
    for col in scored.columns:
        out[f"risk_{col}" if col == "available" else col] = scored[col]
    out["risk_research_only"] = True
    out["risk_policy_status"] = "DIAGNOSTIC_ONLY_UNCALIBRATED"
    return out


def append_risk_metrics(input_path: str, output_path: str) -> dict[str, Any]:
    """Read an OOS CSV and emit a research-only risk-annotated CSV + manifest."""
    source = pd.read_csv(input_path, low_memory=False)
    out = score_frame(source)
    out.to_csv(output_path, index=False)
    available = int(out.get("risk_available", pd.Series(dtype=bool)).fillna(False).sum())
    manifest = {
        "schema_version": 1,
        "status": "PASS" if not out.empty and available == len(out) else "DEFERRED",
        "rows": int(len(out)),
        "risk_available_rows": available,
        "risk_unavailable_rows": int(len(out) - available),
        "research_only": True,
        "probabilities_modified": False,
        "routing_modified": False,
        "promotion_auto": False,
        "threshold_calibration_required": True,
    }
    return manifest
