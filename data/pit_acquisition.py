#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Acquire real NPB/MLB source observations into the PIT ledger.

This collector is intentionally conservative:
- every source response is stored as an immutable, hashed snapshot;
- retrieval time is never masqueraded as a historical announcement time;
- starter names observed without an explicit source announcement timestamp are
  recorded as OBSERVED/UNVERIFIABLE rather than inventing an announcement time;
- failed/unavailable sources are represented explicitly;
- the ledger is append-only JSONL so later PIT replay can reproduce what was
  actually known at each collection cutoff.

Sources:
- MLB Stats API schedule + game feed.
- NPB official site / SPAIA schedule endpoint (the latter is already used by
  the existing NPB multi-source acquisition pipeline).

The collector does not run the expensive historical backtest.
"""
from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import requests

from core.pit_snapshot import append_snapshot, make_snapshot, payload_hash

ROOT = Path(__file__).resolve().parents[1]
PIT_DIR = ROOT / "data" / "pit"
SNAPSHOT_LOG = PIT_DIR / "source_snapshots.jsonl"
EVENT_LOG = PIT_DIR / "event_observations.jsonl"
AVAILABILITY_LOG = PIT_DIR / "availability_observations.jsonl"
RUN_LOG = PIT_DIR / "acquisition_runs.jsonl"

MLB_API = "https://statsapi.mlb.com/api/v1"
NPB_URLS = (
    "https://npb.jp/",
    "https://spaia.jp/baseball/npb/api/weekly_schedule",
)
TIMEOUT = int(os.getenv("PIT_ACQ_TIMEOUT", "30"))
LOOKAHEAD_DAYS = int(os.getenv("PIT_LOOKAHEAD_DAYS", "3"))
LOOKBACK_DAYS = int(os.getenv("PIT_LOOKBACK_DAYS", "1"))

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Baseball-PIT-Acquisition/1.0",
    "Accept": "application/json,text/html;q=0.9,*/*;q=0.8",
})


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=str) + "\n")


def get_json(url: str, params: dict[str, Any] | None = None) -> tuple[Any, str]:
    last: Exception | None = None
    for attempt in range(4):
        try:
            r = SESSION.get(url, params=params, timeout=TIMEOUT)
            r.raise_for_status()
            return r.json(), now_utc()
        except Exception as exc:
            last = exc
            if attempt < 3:
                time.sleep(min(1.5 * (attempt + 1), 5))
    raise RuntimeError(f"request failed: {url}: {last}")


def get_text(url: str) -> tuple[str, str]:
    last: Exception | None = None
    for attempt in range(4):
        try:
            r = SESSION.get(url, timeout=TIMEOUT)
            r.raise_for_status()
            return r.text, now_utc()
        except Exception as exc:
            last = exc
            if attempt < 3:
                time.sleep(min(1.5 * (attempt + 1), 5))
    raise RuntimeError(f"request failed: {url}: {last}")


def _iso_leq(a: str | None, b: str) -> bool:
    if not a:
        return False
    return datetime.fromisoformat(a.replace("Z", "+00:00")) <= datetime.fromisoformat(b.replace("Z", "+00:00"))


def _find_value(obj: Any, names: set[str]) -> Any:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if str(k).lower() in names and v not in (None, "", []):
                return v
        for v in obj.values():
            found = _find_value(v, names)
            if found not in (None, "", []):
                return found
    elif isinstance(obj, list):
        for v in obj:
            found = _find_value(v, names)
            if found not in (None, "", []):
                return found
    return None


def _candidate_games(obj: Any) -> Iterable[dict[str, Any]]:
    """Yield dicts that look like baseball game/event records."""
    if isinstance(obj, dict):
        keys = {str(k).lower() for k in obj}
        gameish = (
            {"gamepk", "gameid", "home", "away"} & keys
            or ("teams" in keys and ("gamepk" in keys or "status" in keys))
            or ("home_team" in keys and "away_team" in keys)
        )
        if gameish:
            yield obj
        for v in obj.values():
            yield from _candidate_games(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _candidate_games(v)


def _mlb_game_id(g: dict[str, Any]) -> str | None:
    for k in ("gamePk", "gamepk", "gameId", "game_id", "id"):
        if g.get(k) not in (None, ""):
            return str(g[k])
    return None


def _mlb_teams(g: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    teams = g.get("teams") if isinstance(g.get("teams"), dict) else {}
    return teams.get("home", {}) or {}, teams.get("away", {}) or {}


def _team_name(side: dict[str, Any]) -> str:
    t = side.get("team") if isinstance(side.get("team"), dict) else {}
    return str(t.get("name") or side.get("name") or "")


def _starter_name(side: dict[str, Any]) -> str | None:
    p = side.get("probablePitcher") or side.get("probable_pitcher")
    if isinstance(p, dict):
        return str(p.get("fullName") or p.get("full_name") or p.get("name") or "") or None
    return None


def _explicit_announcement(g: dict[str, Any], side: str) -> str | None:
    names = {
        f"{side}starterannouncedat", f"{side}_starter_announced_at",
        f"{side}probablepitcherannouncedat", f"{side}_probable_pitcher_announced_at",
        f"{side}pitcherannouncedat", f"{side}_pitcher_announced_at",
    }
    value = _find_value(g, names)
    if value in (None, ""):
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).isoformat()
    except Exception:
        return None


def _record_snapshot(*, event_id: str, league: str, entity_type: str,
                     entity_id: str, source: str, payload: Any,
                     retrieved_at: str, cutoff: str,
                     available_at: str | None, status: str = "KNOWN") -> None:
    # A current observation is available to a prediction whose cutoff is the
    # collection instant. Historical availability is never reconstructed here.
    snap = make_snapshot(
        event_id=event_id, league=league, entity_type=entity_type,
        entity_id=entity_id, source=source, payload=payload,
        prediction_cutoff=cutoff, available_at=available_at,
        source_timestamp=None, retrieved_at=retrieved_at, status=status,
    )
    append_snapshot(snap, SNAPSHOT_LOG)


def acquire_mlb(cutoff: str) -> int:
    start = (datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)).date()
    end = (datetime.now(timezone.utc) + timedelta(days=LOOKAHEAD_DAYS)).date()
    payload, retrieved = get_json(
        f"{MLB_API}/schedule",
        {"sportId": 1, "startDate": str(start), "endDate": str(end),
         "hydrate": "probablePitcher,linescore,venue"},
    )
    _record_snapshot(event_id="MLB-SCHEDULE", league="MLB", entity_type="schedule",
                     entity_id=f"{start}:{end}", source="MLB_STATS_API",
                     payload=payload, retrieved_at=retrieved, cutoff=cutoff,
                     available_at=retrieved)
    count = 0
    seen: set[str] = set()
    for g in _candidate_games(payload):
        gid = _mlb_game_id(g)
        if not gid or gid in seen:
            continue
        seen.add(gid)
        home, away = _mlb_teams(g)
        hname, aname = _team_name(home), _team_name(away)
        hs, ass = _starter_name(home), _starter_name(away)
        row = {
            "event_id": f"MLB:{gid}", "league": "MLB", "game_id": gid,
            "home_team": hname, "away_team": aname,
            "home_starter": hs, "away_starter": ass,
            "home_starter_announced_at": _explicit_announcement(g, "home"),
            "away_starter_announced_at": _explicit_announcement(g, "away"),
            "observed_at": retrieved, "prediction_cutoff": cutoff,
            "source": "MLB_STATS_API", "payload_hash": payload_hash(g),
        }
        _append_jsonl(EVENT_LOG, row)
        _append_jsonl(AVAILABILITY_LOG, {
            **row,
            "starter_status": "ANNOUNCED" if row["home_starter_announced_at"] and row["away_starter_announced_at"] else "OBSERVED_UNVERIFIABLE_ANNOUNCEMENT_TIME",
            "lineup_status": "UNVERIFIABLE",
        })
        _record_snapshot(event_id=f"MLB:{gid}", league="MLB", entity_type="game",
                         entity_id=gid, source="MLB_STATS_API", payload=g,
                         retrieved_at=retrieved, cutoff=cutoff,
                         available_at=retrieved)
        count += 1
    return count


def acquire_npb(cutoff: str) -> int:
    start = (datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)).date()
    end = (datetime.now(timezone.utc) + timedelta(days=LOOKAHEAD_DAYS)).date()
    count = 0
    # Prefer the existing structured SPAIA endpoint. NPB official HTML is kept
    # as an additional source observation even when it is not machine-readable.
    for url in NPB_URLS:
        try:
            if "spaia.jp" in url:
                payload, retrieved = get_json(url)
            else:
                text, retrieved = get_text(url)
                payload = {"url": url, "html_sha256": __import__("hashlib").sha256(text.encode("utf-8", "ignore")).hexdigest()}
            _record_snapshot(event_id="NPB-SCHEDULE", league="NPB", entity_type="schedule",
                             entity_id=f"{start}:{end}:{url}", source=url,
                             payload=payload, retrieved_at=retrieved, cutoff=cutoff,
                             available_at=retrieved)
            for idx, g in enumerate(_candidate_games(payload)):
                gid = _find_value(g, {"gameid", "game_id", "gamepk", "id"})
                if gid is None:
                    gid = f"{payload_hash(g)[:16]}-{idx}"
                gid = str(gid)
                home = _find_value(g, {"home", "home_team", "hometeam"})
                away = _find_value(g, {"away", "away_team", "awayteam"})
                if isinstance(home, dict):
                    home = home.get("name") or home.get("team")
                if isinstance(away, dict):
                    away = away.get("name") or away.get("team")
                if not home or not away:
                    continue
                row = {
                    "event_id": f"NPB:{gid}", "league": "NPB", "game_id": gid,
                    "home_team": str(home), "away_team": str(away),
                    "home_starter": _find_value(g, {"homestarter", "home_starter", "homepitcher"}),
                    "away_starter": _find_value(g, {"awaystarter", "away_starter", "awaypitcher"}),
                    "home_starter_announced_at": _explicit_announcement(g, "home"),
                    "away_starter_announced_at": _explicit_announcement(g, "away"),
                    "observed_at": retrieved, "prediction_cutoff": cutoff,
                    "source": url, "payload_hash": payload_hash(g),
                }
                _append_jsonl(EVENT_LOG, row)
                _append_jsonl(AVAILABILITY_LOG, {
                    **row,
                    "starter_status": "ANNOUNCED" if row["home_starter_announced_at"] and row["away_starter_announced_at"] else "OBSERVED_UNVERIFIABLE_ANNOUNCEMENT_TIME",
                    "lineup_status": "UNVERIFIABLE",
                })
                _record_snapshot(event_id=f"NPB:{gid}", league="NPB", entity_type="game",
                                 entity_id=gid, source=url, payload=g,
                                 retrieved_at=retrieved, cutoff=cutoff,
                                 available_at=retrieved)
                count += 1
        except Exception as exc:
            _record_snapshot(event_id="NPB-SCHEDULE", league="NPB", entity_type="schedule",
                             entity_id=f"{start}:{end}:{url}", source=url,
                             payload={"error": str(exc)}, retrieved_at=now_utc(),
                             cutoff=cutoff, available_at=None, status="UNAVAILABLE")
            print(f"[PIT][NPB] source unavailable: {url}: {exc}")
    return count


def main() -> None:
    PIT_DIR.mkdir(parents=True, exist_ok=True)
    cutoff = now_utc()
    started = cutoff
    results: dict[str, Any] = {"run_started_at": started, "cutoff": cutoff}
    for league, fn in (("MLB", acquire_mlb), ("NPB", acquire_npb)):
        try:
            results[f"{league.lower()}_events"] = fn(cutoff)
            results[f"{league.lower()}_status"] = "OK"
        except Exception as exc:
            results[f"{league.lower()}_events"] = 0
            results[f"{league.lower()}_status"] = "UNAVAILABLE"
            results[f"{league.lower()}_error"] = str(exc)
            print(f"[PIT][{league}] acquisition failed: {exc}")
    results["run_finished_at"] = now_utc()
    _append_jsonl(RUN_LOG, results)
    if results.get("mlb_status") != "OK" and results.get("npb_status") != "OK":
        raise SystemExit("Both NPB and MLB PIT sources failed")
    print(json.dumps(results, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
