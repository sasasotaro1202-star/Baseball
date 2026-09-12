"""Production prediction eligibility gate and canonical logging adapter."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Mapping

from core.pit import _ts
from data.availability import AvailabilityRecord, prediction_eligible
from prediction.prediction_log import PredictionRecord, append_prediction, make_prediction_id


def eligibility_gate(*, availability: AvailabilityRecord, required_data_ok: bool,
                     feature_complete: bool, model_available: bool,
                     calibration_available: bool) -> tuple[bool, list[str]]:
    ok, reasons = prediction_eligible(availability)
    if not required_data_ok:
        reasons.append("required_data_unavailable")
    if not feature_complete:
        reasons.append("feature_incomplete")
    if not model_available:
        reasons.append("model_unavailable")
    if not calibration_available:
        reasons.append("calibration_unavailable")
    cutoff = _ts(availability.prediction_cutoff)
    if cutoff > _ts(availability.retrieved_at):
        # Retrieval may happen after cutoff only for historical replay; production
        # callers should supply a cutoff at or before retrieval.
        reasons.append("cutoff_after_retrieval")
    return not reasons, reasons


def run_prediction(*, row: Mapping[str, Any], availability: AvailabilityRecord,
                   probability_fn: Callable[[Mapping[str, Any]], Mapping[str, float]],
                   model_version: str, feature_version: str, calibration_version: str,
                   git_commit: str, data_snapshot_id: str, log_path: str) -> dict[str, Any]:
    """Run the final gate, score callback and immutable prediction ledger write.

    The callback is intentionally injected: this adapter never trains or
    fabricates a model. Existing BaseballBacktest/production code remains the
    source of numerical predictions.
    """
    eligible, reasons = eligibility_gate(
        availability=availability,
        required_data_ok=bool(row.get("required_data_ok", True)),
        feature_complete=bool(row.get("feature_complete", True)),
        model_available=bool(row.get("model_available", True)),
        calibration_available=bool(row.get("calibration_available", True)),
    )
    if not eligible:
        return {"eligible": False, "reasons": reasons, "event_id": availability.event_id}

    probabilities = dict(probability_fn(row))
    expected = {"home", "away"} | ({"draw"} if availability.league == "NPB" else set())
    if set(probabilities) != expected:
        raise ValueError("probability callback returned the wrong league contract")
    total = sum(float(v) for v in probabilities.values())
    if abs(total - 1.0) > 1e-8:
        raise ValueError("final probabilities must sum to 1")

    now = datetime.now(timezone.utc).isoformat()
    pid = make_prediction_id(availability.event_id, availability.prediction_cutoff, model_version, git_commit)
    record = PredictionRecord(
        prediction_id=pid, event_id=availability.event_id, league=availability.league,
        prediction_cutoff=availability.prediction_cutoff, prediction_created_at=now,
        home_team=availability.home_team, away_team=availability.away_team,
        home_starter=availability.home_starter, away_starter=availability.away_starter,
        probabilities={k: float(v) for k, v in probabilities.items()},
        score_candidates=list(row.get("score_candidates", [])),
        low_probability=row.get("low_probability"), high_probability=row.get("high_probability"),
        total_runs_line=row.get("total_runs_line"), confidence=row.get("confidence"),
        volatility=row.get("volatility"), model_version=model_version,
        feature_version=feature_version, calibration_version=calibration_version,
        git_commit=git_commit, data_snapshot_id=data_snapshot_id,
    )
    append_prediction(record, log_path)
    return {"eligible": True, "prediction": record}
