"""Point-in-time market-line contract."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math


@dataclass(frozen=True)
class TotalRunsLine:
    event_id: str
    league: str
    line: float
    source: str
    observed_at: str
    available_at: str
    status: str = "KNOWN"

    def validate(self) -> None:
        if not str(self.event_id).strip():
            raise ValueError("event_id is required")
        if not str(self.league).strip():
            raise ValueError("league is required")
        value = float(self.line)
        if not math.isfinite(value) or value < 0 or abs(value * 2 - round(value * 2)) > 1e-9:
            raise ValueError("line must be finite, non-negative, integer or half-point")
        if not str(self.source).strip():
            raise ValueError("source is required")
        for name, raw in (("observed_at", self.observed_at), ("available_at", self.available_at)):
            try:
                datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
            except Exception as exc:
                raise ValueError(f"{name} must be an ISO-8601 timestamp") from exc
        if self.status not in {"KNOWN", "UNKNOWN"}:
            raise ValueError("status must be KNOWN or UNKNOWN")

    def pit_usable(self, cutoff: str) -> bool:
        self.validate()
        try:
            available = datetime.fromisoformat(str(self.available_at).replace("Z", "+00:00"))
            limit = datetime.fromisoformat(str(cutoff).replace("Z", "+00:00"))
        except Exception as exc:
            raise ValueError("cutoff must be an ISO-8601 timestamp") from exc
        if available.tzinfo is None:
            available = available.replace(tzinfo=timezone.utc)
        if limit.tzinfo is None:
            limit = limit.replace(tzinfo=timezone.utc)
        return self.status == "KNOWN" and available <= limit
