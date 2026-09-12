"""PIT snapshot/provenance contract for Baseball data.

A snapshot is an immutable record of what was known, when it was retrieved,
and when the source said it became available. This module does not infer
availability from retrieval time.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class SourceSnapshot:
    event_id: str
    league: str
    entity_type: str
    entity_id: str
    source: str
    source_timestamp: str | None
    retrieved_at: str
    available_at: str | None
    prediction_cutoff: str
    payload_hash: str
    status: str = "KNOWN"

    def validate(self) -> None:
        if not self.event_id or not self.entity_id or not self.source:
            raise ValueError("event_id, entity_id and source are required")
        if self.status not in {"KNOWN", "MISSING", "UNAVAILABLE", "UNVERIFIABLE"}:
            raise ValueError(f"invalid snapshot status: {self.status}")
        if self.available_at and self.prediction_cutoff:
            a = datetime.fromisoformat(self.available_at.replace("Z", "+00:00"))
            c = datetime.fromisoformat(self.prediction_cutoff.replace("Z", "+00:00"))
            if a > c:
                raise ValueError("PIT violation: availability is after prediction cutoff")


def payload_hash(payload: Any) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def make_snapshot(*, event_id: str, league: str, entity_type: str, entity_id: str,
                  source: str, payload: Any, prediction_cutoff: str,
                  available_at: str | None, source_timestamp: str | None = None,
                  retrieved_at: str | None = None, status: str = "KNOWN") -> SourceSnapshot:
    snap = SourceSnapshot(
        event_id=str(event_id), league=league, entity_type=entity_type,
        entity_id=str(entity_id), source=source,
        source_timestamp=source_timestamp,
        retrieved_at=retrieved_at or datetime.now(timezone.utc).isoformat(),
        available_at=available_at, prediction_cutoff=prediction_cutoff,
        payload_hash=payload_hash(payload), status=status,
    )
    snap.validate()
    return snap


def append_snapshot(snapshot: SourceSnapshot, path: str | Path) -> None:
    """Append an auditable JSONL snapshot; existing records are never rewritten."""
    snapshot.validate()
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(asdict(snapshot), ensure_ascii=False, sort_keys=True) + "\n")


def validate_snapshot_rows(rows: list[Mapping[str, Any]]) -> dict[str, int]:
    counts = {k: 0 for k in ("KNOWN", "MISSING", "UNAVAILABLE", "UNVERIFIABLE")}
    for row in rows:
        status = str(row.get("status", "KNOWN"))
        if status not in counts:
            raise ValueError(f"invalid snapshot status: {status}")
        counts[status] += 1
    return counts
