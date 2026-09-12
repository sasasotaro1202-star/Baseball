"""Immutable prediction ledger for Baseball production outputs."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class PredictionRecord:
    prediction_id: str
    event_id: str
    league: str
    prediction_cutoff: str
    prediction_created_at: str
    home_team: str
    away_team: str
    home_starter: str | None
    away_starter: str | None
    probabilities: dict[str, float]
    score_candidates: list[dict[str, Any]]
    low_probability: float | None
    high_probability: float | None
    total_runs_line: float | None
    confidence: float | None
    volatility: float | None
    model_version: str
    feature_version: str
    calibration_version: str
    git_commit: str
    data_snapshot_id: str
    eligibility: str = "ELIGIBLE"


def make_prediction_id(event_id: str, cutoff: str, model_version: str, git_commit: str) -> str:
    return hashlib.sha256(f"{event_id}|{cutoff}|{model_version}|{git_commit}".encode()).hexdigest()[:24]


def validate_prediction(record: PredictionRecord) -> None:
    if record.league not in {"NPB", "MLB"}:
        raise ValueError("league must be NPB or MLB")
    if record.eligibility != "ELIGIBLE":
        raise ValueError("only eligible predictions may enter the production ledger")
    if record.home_starter is None or record.away_starter is None:
        raise ValueError("both starting pitchers must be confirmed")
    required = {"home", "away"} | ({"draw"} if record.league == "NPB" else set())
    if set(record.probabilities) != required:
        raise ValueError("probability contract does not match league")
    total = sum(float(v) for v in record.probabilities.values())
    if abs(total - 1.0) > 1e-8:
        raise ValueError("prediction probabilities must sum to 1")


def append_prediction(record: PredictionRecord, path: str | Path) -> None:
    validate_prediction(record)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(asdict(record), ensure_ascii=False, sort_keys=True) + "\n")


def record_from_mapping(row: Mapping[str, Any]) -> PredictionRecord:
    record = PredictionRecord(**dict(row))
    validate_prediction(record)
    return record
