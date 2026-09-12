"""Historical market-line contract for PIT-safe Low/High evaluation."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Mapping


@dataclass(frozen=True)
class TotalRunsLine:
    event_id: str
    league: str
    line: float | None
    source: str
    observed_at: str
    available_at: str | None
    status: str = "KNOWN"

    def validate(self) -> None:
        if self.status not in {"KNOWN", "MISSING", "UNAVAILABLE", "UNVERIFIABLE"}:
            raise ValueError(f"invalid line status: {self.status}")
        if self.status == "KNOWN" and self.line is None:
            raise ValueError("KNOWN market line requires a numeric line")
        if self.line is not None and self.line < 0:
            raise ValueError("total runs line cannot be negative")


def low_high_threshold(line: float) -> tuple[float, float]:
    """Return the integer boundary for a standard .5 total line."""
    if abs(float(line) * 2 - round(float(line) * 2)) > 1e-9:
        raise ValueError("market line must be numeric")
    cutoff = int(float(line) // 1) if float(line).is_integer() else int(float(line) + 0.5)
    return float(cutoff), float(cutoff + 1)


def classify_total(total_runs: int, line: float) -> str:
    if float(line).is_integer():
        if total_runs == int(line):
            return "PUSH"
        return "LOW" if total_runs < line else "HIGH"
    return "LOW" if total_runs < line else "HIGH"


def from_mapping(row: Mapping[str, Any]) -> TotalRunsLine:
    line = row.get("line")
    obj = TotalRunsLine(
        event_id=str(row["event_id"]), league=str(row["league"]),
        line=None if line in (None, "") else float(line),
        source=str(row.get("source", "UNVERIFIABLE")),
        observed_at=str(row["observed_at"]),
        available_at=row.get("available_at"), status=str(row.get("status", "KNOWN")),
    )
    obj.validate()
    return obj


def pit_usable(line: TotalRunsLine, cutoff: str) -> bool:
    line.validate()
    if line.status != "KNOWN" or line.line is None or not line.available_at:
        return False
    return datetime.fromisoformat(line.available_at.replace("Z", "+00:00")) <= datetime.fromisoformat(cutoff.replace("Z", "+00:00"))
