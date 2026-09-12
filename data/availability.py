"""PIT-aware starter/lineup announcement records and eligibility gates."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Mapping


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
        if self.home_starter and not self.home_starter_announced_at:
            raise ValueError("home starter without announcement timestamp")
        if self.away_starter and not self.away_starter_announced_at:
            raise ValueError("away starter without announcement timestamp")
        cutoff = datetime.fromisoformat(self.prediction_cutoff.replace("Z", "+00:00"))
        for value in (self.home_starter_announced_at, self.away_starter_announced_at, self.lineup_announced_at):
            if value:
                if datetime.fromisoformat(value.replace("Z", "+00:00")) > cutoff:
                    raise ValueError("availability timestamp is after prediction cutoff")


def from_mapping(row: Mapping[str, Any]) -> AvailabilityRecord:
    record = AvailabilityRecord(
        event_id=str(row["event_id"]), league=str(row["league"]),
        home_team=str(row["home_team"]), away_team=str(row["away_team"]),
        home_starter=row.get("home_starter"), away_starter=row.get("away_starter"),
        home_starter_announced_at=row.get("home_starter_announced_at"),
        away_starter_announced_at=row.get("away_starter_announced_at"),
        lineup_status=str(row.get("lineup_status", "UNVERIFIABLE")),
        lineup_announced_at=row.get("lineup_announced_at"),
        source=str(row.get("source", "UNVERIFIABLE")),
        retrieved_at=str(row["retrieved_at"]),
        prediction_cutoff=str(row["prediction_cutoff"]),
    )
    record.validate()
    return record


def prediction_eligible(record: AvailabilityRecord) -> tuple[bool, list[str]]:
    record.validate()
    reasons: list[str] = []
    if not record.home_starter:
        reasons.append("home_starter_not_confirmed")
    if not record.away_starter:
        reasons.append("away_starter_not_confirmed")
    if record.lineup_status == "UNAVAILABLE":
        reasons.append("lineup_unavailable")
    return (not reasons, reasons)


def as_dict(record: AvailabilityRecord) -> dict[str, Any]:
    return asdict(record)
