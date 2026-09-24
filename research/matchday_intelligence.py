#!/usr/bin/env python3
"""Research-only point-in-time Matchday Intelligence layer.

The layer stores pregame observations separately from the historical model.
No observation can affect a forecast unless its availability timestamp is at or
before the prediction timestamp. Missing/unknown context is preserved as
UNKNOWN rather than converted to a neutral numeric value silently.

This module intentionally does not contain learned adjustment magnitudes.
Candidate context effects must be estimated and accepted by chronological OOS
replay before they can affect a production forecast.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional


class ContextState(str, Enum):
    VERIFIED = "VERIFIED"
    PROJECTED = "PROJECTED"
    STALE = "STALE"
    CONFLICT = "CONFLICT"
    UNKNOWN = "UNKNOWN"


class ContextKind(str, Enum):
    STARTER = "STARTER"
    LINEUP = "LINEUP"
    INJURY = "INJURY"
    AVAILABILITY = "AVAILABILITY"
    REST_TRAVEL = "REST_TRAVEL"
    WEATHER = "WEATHER"
    MARKET = "MARKET"
    BULLPEN = "BULLPEN"


class ContextEvent(str, Enum):
    STARTER_CONFIRMED = "STARTER_CONFIRMED"
    STARTER_CHANGED = "STARTER_CHANGED"
    LINEUP_PROJECTED = "LINEUP_PROJECTED"
    LINEUP_CONFIRMED = "LINEUP_CONFIRMED"
    LINEUP_CHANGED = "LINEUP_CHANGED"
    PLAYER_OUT = "PLAYER_OUT"
    PLAYER_RETURNED = "PLAYER_RETURNED"
    WEATHER_CHANGED = "WEATHER_CHANGED"
    REST_ASYMMETRY = "REST_ASYMMETRY"
    TRAVEL_BURDEN = "TRAVEL_BURDEN"
    MARKET_MOVED = "MARKET_MOVED"
    BULLPEN_STATE_CHANGED = "BULLPEN_STATE_CHANGED"


@dataclass(frozen=True)
class Observation:
    game_id: str
    snapshot_id: str
    prediction_time: str
    available_at: str
    source_time: Optional[str]
    retrieved_at: str
    kind: str
    state: str
    source: str
    value: Any
    freshness_seconds: Optional[float] = None
    confidence: Optional[float] = None


def _parse_ts(value: str) -> datetime:
    dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def is_pit_safe(observation: Observation) -> bool:
    """Strict PIT predicate: unknown availability is never considered safe."""
    try:
        return _parse_ts(observation.available_at) <= _parse_ts(observation.prediction_time)
    except Exception:
        return False


def usable_observations(
    observations: Iterable[Observation],
    prediction_time: str,
) -> List[Observation]:
    """Return only observations provably available by prediction_time."""
    cutoff = _parse_ts(prediction_time)
    out = []
    for obs in observations:
        try:
            if _parse_ts(obs.available_at) <= cutoff and _parse_ts(obs.prediction_time) == cutoff:
                out.append(obs)
        except Exception:
            continue
    return out


def build_observation(
    *,
    game_id: str,
    snapshot_id: str,
    prediction_time: str,
    available_at: str,
    kind: str,
    state: str,
    source: str,
    value: Any,
    source_time: Optional[str] = None,
    retrieved_at: Optional[str] = None,
    freshness_seconds: Optional[float] = None,
    confidence: Optional[float] = None,
) -> Observation:
    retrieved = retrieved_at or datetime.now(timezone.utc).isoformat()
    obs = Observation(
        game_id=str(game_id),
        snapshot_id=str(snapshot_id),
        prediction_time=str(prediction_time),
        available_at=str(available_at),
        source_time=source_time,
        retrieved_at=retrieved,
        kind=str(kind),
        state=str(state),
        source=str(source),
        value=value,
        freshness_seconds=freshness_seconds,
        confidence=confidence,
    )
    if not is_pit_safe(obs):
        raise ValueError("MATCHDAY_PIT_FAIL: observation was not available by prediction_time")
    return obs


def observation_delta_events(
    previous: Optional[Observation],
    current: Observation,
) -> List[str]:
    """Emit explainable context-change events without using target outcomes."""
    if not is_pit_safe(current):
        raise ValueError("MATCHDAY_PIT_FAIL: current observation is not PIT-safe")
    events: List[str] = []
    kind = current.kind.upper()
    old_state = previous.state.upper() if previous else None
    new_state = current.state.upper()

    if kind == ContextKind.STARTER.value:
        if new_state == ContextState.VERIFIED.value and old_state != new_state:
            events.append(ContextEvent.STARTER_CONFIRMED.value)
        if previous is not None and previous.value != current.value:
            events.append(ContextEvent.STARTER_CHANGED.value)
    elif kind == ContextKind.LINEUP.value:
        if new_state == ContextState.PROJECTED.value and old_state != new_state:
            events.append(ContextEvent.LINEUP_PROJECTED.value)
        if new_state == ContextState.VERIFIED.value and old_state != new_state:
            events.append(ContextEvent.LINEUP_CONFIRMED.value)
    elif kind == ContextKind.INJURY.value:
        before = str(previous.value).upper() if previous is not None else ""
        after = str(current.value).upper()
        if after in {"OUT", "IL", "SCRATCH"} and before not in {"OUT", "IL", "SCRATCH"}:
            events.append(ContextEvent.PLAYER_OUT.value)
        if after in {"ACTIVE", "RETURNED"} and before in {"OUT", "IL", "SCRATCH"}:
            events.append(ContextEvent.PLAYER_RETURNED.value)
    elif kind == ContextKind.WEATHER.value and previous is not None and previous.value != current.value:
        events.append(ContextEvent.WEATHER_CHANGED.value)
    elif kind == ContextKind.MARKET.value and previous is not None and previous.value != current.value:
        events.append(ContextEvent.MARKET_MOVED.value)
    elif kind == ContextKind.REST_TRAVEL.value:
        current_value = current.value if isinstance(current.value, dict) else {}
        if isinstance(current_value, dict):
            h = current_value.get("home", {}) if isinstance(current_value.get("home"), dict) else {}
            a = current_value.get("away", {}) if isinstance(current_value.get("away"), dict) else {}
            try:
                if abs(float(h.get("rest_days", 0.0)) - float(a.get("rest_days", 0.0))) > 1e-9:
                    events.append(ContextEvent.REST_ASYMMETRY.value)
            except Exception:
                pass
    elif kind == ContextKind.AVAILABILITY.value:
        before = str(previous.value).upper() if previous is not None else ""
        after = str(current.value).upper()
        if after == ContextEvent.PLAYER_OUT.value and before != ContextEvent.PLAYER_OUT.value:
            events.append(ContextEvent.PLAYER_OUT.value)
        if after == ContextEvent.PLAYER_RETURNED.value and before != ContextEvent.PLAYER_RETURNED.value:
            events.append(ContextEvent.PLAYER_RETURNED.value)
    elif kind == ContextKind.BULLPEN.value and previous is not None and previous.value != current.value:
        events.append(ContextEvent.BULLPEN_STATE_CHANGED.value)

    return events


def snapshot_summary(observations: Iterable[Observation], prediction_time: str) -> Dict[str, Any]:
    rows = usable_observations(observations, prediction_time)
    counts = {state.value: 0 for state in ContextState}
    kinds = {kind.value: 0 for kind in ContextKind}
    for row in rows:
        counts[row.state] = counts.get(row.state, 0) + 1
        kinds[row.kind] = kinds.get(row.kind, 0) + 1
    return {
        "prediction_time": str(prediction_time),
        "observation_count": len(rows),
        "state_counts": counts,
        "kind_counts": kinds,
        "pit_safe": True,
    }


def self_test() -> Dict[str, Any]:
    t = "2026-09-24T08:00:00+00:00"
    starter = build_observation(
        game_id="g1",
        snapshot_id="s1",
        prediction_time=t,
        available_at="2026-09-24T07:00:00+00:00",
        kind=ContextKind.STARTER.value,
        state=ContextState.PROJECTED.value,
        source="NPB_OFFICIAL",
        value="starter-a",
        source_time="2026-09-24T06:55:00+00:00",
    )
    confirmed = build_observation(
        game_id="g1",
        snapshot_id="s2",
        prediction_time=t,
        available_at="2026-09-24T07:45:00+00:00",
        kind=ContextKind.STARTER.value,
        state=ContextState.VERIFIED.value,
        source="NPB_OFFICIAL",
        value="starter-b",
        source_time="2026-09-24T07:44:00+00:00",
    )
    assert is_pit_safe(confirmed)
    assert ContextEvent.STARTER_CONFIRMED.value in observation_delta_events(starter, confirmed)
    assert ContextEvent.STARTER_CHANGED.value in observation_delta_events(starter, confirmed)

    try:
        build_observation(
            game_id="g1",
            snapshot_id="future",
            prediction_time=t,
            available_at="2026-09-24T08:01:00+00:00",
            kind=ContextKind.WEATHER.value,
            state=ContextState.VERIFIED.value,
            source="WEATHER",
            value={"wind_kmh": 20},
        )
    except ValueError:
        pass
    else:
        raise AssertionError("future observation must fail closed")

    summary = snapshot_summary([starter, confirmed], t)
    assert summary["pit_safe"] is True
    assert summary["observation_count"] == 2
    return {"status": "PASS", "summary": summary}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), ensure_ascii=False, sort_keys=True))
