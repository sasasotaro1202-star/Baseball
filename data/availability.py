"""PIT-aware starter/lineup announcement records and eligibility gates.

The key invariant is that a starter is not considered PIT-confirmed merely
because the name is known.  The announcement time must also be known and must
be at or before the prediction cutoff.  Unknown announcement timing therefore
fails closed for production prediction.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Mapping


def _dt(value: str) -> datetime:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


@dataclass(frozen=True)
class AvailabilityRecord:
    event_id: str
    league: str
    home_team: str
    away_team: str
    home_starter: str | None
    away_starter: str | None
    home_starter_announced_at: str | None
    away_starter_announced_at: str | None
    lineup_status: str
    lineup_announced_at: str | None
    source: str
    retrieved_at: str
    prediction_cutoff: str

    def validate(self) -> None:
        if not self.event_id or self.league not in {"NPB", "MLB"}:
            raise ValueError("event_id and league (NPB/MLB) are required")
        if not self.home_team or not self.away_team:
            raise ValueError("home_team and away_team are required")

        cutoff = _dt(self.prediction_cutoff)
        retrieved = _dt(self.retrieved_at)
        if cutoff.tzinfo is None or retrieved.tzinfo is None:
            raise ValueError("retrieved_at and prediction_cutoff must be timezone-aware")
        # A source observation cannot be used at a cutoff earlier than the
        # retrieval time.  Retrieval at or before cutoff is PIT-safe.
        if retrieved > cutoff:
            raise ValueError("retrieved_at is after prediction cutoff")

        for starter, announced_at, side in (
            (self.home_starter, self.home_starter_announced_at, "home"),
            (self.away_starter, self.away_starter_announced_at, "away"),
        ):
            if starter and not announced_at:
                raise ValueError(f"{side} starter has unknown announcement timestamp")
            if announced_at:
                ts = _dt(announced_at)
                if ts.tzinfo is None:
                    raise ValueError(f"{side} starter announcement timestamp must be timezone-aware")
                if ts > cutoff:
                    raise ValueError(f"{side} starter was announced after prediction cutoff")
                if ts > retrieved:
                    raise ValueError(f"{side} starter announcement is after source retrieval")

        if self.lineup_announced_at:
            ts = _dt(self.lineup_announced_at)
            if ts.tzinfo is None:
                raise ValueError("lineup announcement timestamp must be timezone-aware")
            if ts > cutoff or ts > retrieved:
                raise ValueError("lineup announcement is not PIT-safe")


def from_mapping(row: Mapping[str, Any]) -> AvailabilityRecord:
    record = AvailabilityRecord(
        event_id=str(row["event_id"]),
        league=str(row["league"]),
        home_team=str(row["home_team"]),
        away_team=str(row["away_team"]),
        home_starter=row.get("home_starter"),
        away_starter=row.get("away_starter"),
        home_starter_announced_at=row.get("home_starter_announced_at"),
        away_starter_announced_at=row.get("away_starter_announced_at"),
        lineup_status=str(row.get("lineup_status", "UNVERIFIABLE")).upper(),
        lineup_announced_at=row.get("lineup_announced_at"),
        source=str(row.get("source", "UNVERIFIABLE")),
        retrieved_at=str(row["retrieved_at"]),
        prediction_cutoff=str(row["prediction_cutoff"]),
    )
    record.validate()
    return record


def prediction_eligible(record: AvailabilityRecord) -> tuple[bool, list[str]]:
    """Fail closed unless both starters are PIT-confirmed.

    Lineup availability is intentionally not a universal blocker: some valid
    pregame markets are published before official lineups.  Callers that require
    lineup data must enforce that separately via ``required_data_ok``.
    """
    record.validate()
    reasons: list[str] = []
    if not record.home_starter:
        reasons.append("home_starter_not_confirmed")
    elif not record.home_starter_announced_at:
        reasons.append("home_starter_announcement_time_unverified")
    if not record.away_starter:
        reasons.append("away_starter_not_confirmed")
    elif not record.away_starter_announced_at:
        reasons.append("away_starter_announcement_time_unverified")
    return (not reasons, reasons)


def as_dict(record: AvailabilityRecord) -> dict[str, Any]:
    return asdict(record)
