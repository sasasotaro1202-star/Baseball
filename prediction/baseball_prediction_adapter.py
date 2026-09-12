"""Bridge the existing BaseballBacktest outputs into the production runner contract.

This is an opt-in wrapper: the legacy numerical engine remains the source of
truth for model probabilities and expected runs. The adapter adds strict output
contracts, exact top-4 score cells, and PIT-safe Low/High handling when a
verified total-runs line is supplied.
"""
from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from data.market_lines import TotalRunsLine
from prediction.score_contract import validate_low_high


def _poisson_pmf(k: int, lam: float) -> float:
    lam = max(float(lam), 1e-9)
    return math.exp(-lam + k * math.log(lam) - math.lgamma(k + 1))


def _score_distribution(lam_home: float, lam_away: float, max_runs: int = 20) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not math.isfinite(lam_home) or not math.isfinite(lam_away) or lam_home <= 0 or lam_away <= 0:
        raise ValueError("expected runs must be finite and positive")
    h = np.array([_poisson_pmf(k, lam_home) for k in range(max_runs + 1)])
    a = np.array([_poisson_pmf(k, lam_away) for k in range(max_runs + 1)])
    joint = np.outer(h, a)
    joint /= joint.sum()
    return h, a, joint


def build_score_candidates(lam_home: float, lam_away: float, *, n: int = 4) -> list[dict[str, Any]]:
    """Return exactly n highest-probability exact score cells."""
    if n < 1:
        raise ValueError("n must be positive")
    _, _, joint = _score_distribution(lam_home, lam_away)
    cells = [(int(h), int(a), float(joint[h, a])) for h in range(joint.shape[0]) for a in range(joint.shape[1])]
    cells.sort(key=lambda x: (-x[2], x[0], x[1]))
    return [{"score": f"{h}-{a}", "probability": p} for h, a, p in cells[:n]]


def build_low_high(lam_home: float, lam_away: float, line: TotalRunsLine | None) -> tuple[float | None, float | None]:
    """Return Low/High only when a verified PIT-usable market line exists."""
    if line is None:
        return None, None
    # The caller is responsible for supplying the correct prediction cutoff to
    # line.pit_usable(). This function refuses a non-KNOWN line regardless.
    line.validate()
    if line.status != "KNOWN" or line.line is None:
        return None, None
    _, _, joint = _score_distribution(lam_home, lam_away)
    total = np.add.outer(np.arange(joint.shape[0]), np.arange(joint.shape[1]))
    if line.line.is_integer():
        low = float(joint[total < line.line].sum())
        # Integer lines have PUSH mass; production runner must not pretend that
        # a two-way Low/High market exists unless the supplied market contract
        # explicitly defines how Push is handled. Return no two-way probability.
        return None, None
    low = float(joint[total < line.line].sum())
    high = float(1.0 - low)
    return validate_low_high(low, high)


def build_runner_row(*, probabilities: Mapping[str, float], lam_home: float, lam_away: float,
                      total_line: TotalRunsLine | None = None,
                      cutoff: str | None = None) -> dict[str, Any]:
    """Build the non-identity portion of a canonical runner input row."""
    out = {"score_candidates": build_score_candidates(lam_home, lam_away, n=4)}
    if total_line is not None and cutoff is not None and not total_line.pit_usable(cutoff):
        raise ValueError("total-runs line is not PIT-usable at the prediction cutoff")
    low, high = build_low_high(lam_home, lam_away, total_line)
    out.update({"low_probability": low, "high_probability": high,
                "total_runs_line": None if total_line is None else total_line.line})
    # Keep the league-specific probability mapping unchanged; runner validates
    # its exact Home/Away or Home/Draw/Away contract.
    out["probabilities"] = dict(probabilities)
    return out
