#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Full-scope, incremental MLB source acquisition.

The dataset scope is NOT reduced: the first bootstrap acquires the full
historical window in resumable date chunks, then subsequent runs refresh only
the delta while retaining all historical rows locally.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

API = "https://statsapi.mlb.com/api/v1"
DATA = Path(os.getenv("BASEBALL_DATA_DIR", "data"))
CACHE = DATA / "mlb_games.csv"
CHECKPOINTS = DATA / "checkpoints"
CHUNK_DIR = CHECKPOINTS / "mlb_schedule_chunks"
MANIFEST = CHECKPOINTS / "mlb_acquisition_manifest.json"
TIMEOUT = int(os.getenv("MLB_INCREMENTAL_TIMEOUT", "20"))
CORRECTION_DAYS = int(os.getenv("MLB_CORRECTION_DAYS", "7"))
BOOTSTRAP_START_YEAR = int(os.getenv("MLB_BOOTSTRAP_START_YEAR", "2020"))
CHUNK_DAYS = max(14, int(os.getenv("MLB_BOOTSTRAP_CHUNK_DAYS", "45")))
TODAY = datetime.now(timezone.utc).date()
SEASON = TODAY.year

S = requests.Session()
S.headers.update({"User-Agent": "BaseballIncrementalAcquisition/1.2", "Accept": "application/json"})


def get_json(url, params=None):
    last = None
    for attempt in range(5):
        try:
            r = S.get(url, params=params, timeout=TIMEOUT)
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            last = exc
            if attempt == 4:
                raise
            time.sleep(min(1.5 * (attempt + 1), 8))
    raise RuntimeError(last)


def norm(df):
    if df.empty:
        return df
    for c in ["home_score", "away_score"]:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
    df["game_id"] = df["game_id"].astype(str)
    df = df.dropna(subset=["datetime", "home_score", "away_score", "home", "away"])
    return df.sort_values(["datetime", "game_id"]).drop_duplicates("game_id", keep="last").reset_index(drop=True)


def atomic_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def chunk_path(start_date, end_date) -> Path:
    return CHUNK_DIR / f"{start_date:%Y%m%d}_{end_date:%Y%m%d}.csv"


EMPTY_SCHEMA = [
    "league", "game_id", "datetime", "home", "away", "home_score", "away_score",
    "home_starter", "away_starter", "venue", "game_type", "series_description",
    "confirmed_starters",
]


def empty_chunk_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=EMPTY_SCHEMA)


def read_existing_cache():
    if CACHE.exists():
        try:
            return norm(pd.read_csv(CACHE, low_memory=False))
        except Exception as exc:
            raise RuntimeError(f"MLB cache exists but is unreadable; refusing to discard it: {exc}") from exc
    return pd.DataFrame()


def fetch_schedule(start_date, end_date):
    data = get_json(f"{API}/schedule", {
        "sportId": 1,
        "startDate": str(start_date),
        "endDate": str(end_date),
        "hydrate": "probablePitcher,linescore",
    })
    rows = []
    for block in data.get("dates", []):
        for game in block.get("games", []):
            if game.get("status", {}).get("abstractGameState") != "Final":
                continue
            teams = game.get("teams", {})
            home = teams.get("home", {})
            away = teams.get("away", {})
            rows.append({
                "league": "MLB",
                "game_id": str(game.get("gamePk")),
                "datetime": game.get("gameDate"),
                "home": home.get("team", {}).get("name", ""),
                "away": away.get("team", {}).get("name", ""),
                "home_score": home.get("score"),
                "away_score": away.get("score"),
                "home_starter": (home.get("probablePitcher") or {}).get("fullName", ""),
                "away_starter": (away.get("probablePitcher") or {}).get("fullName", ""),
                "venue": (game.get("venue") or {}).get("name", ""),
                "game_type": str(game.get("gameType") or "").strip(),
                "series_description": str(game.get("seriesDescription") or "").strip(),
                "confirmed_starters": bool(
                    (home.get("probablePitcher") or {}).get("fullName")
                    and (away.get("probablePitcher") or {}).get("fullName")
                ),
            })
    return pd.DataFrame(rows)


def acquire_chunked(start_date, end_date):
    frames = []
    completed = []
    cursor = start_date
    while cursor <= end_date:
        chunk_end = min(end_date, cursor + timedelta(days=CHUNK_DAYS - 1))
        path = chunk_path(cursor, chunk_end)
        refetch = not path.exists()
        if path.exists():
            # Classify the raw checkpoint bytes first. This avoids pandas parser
            # behavior deciding whether a deliberately empty range is valid.
            raw = path.read_bytes()
            if raw == b"":
                refetch = True
                reason = "zero-byte checkpoint"
            elif raw == b"\n":
                # pandas writes an empty DataFrame as a single newline. It is a
                # valid completed no-games checkpoint, not corruption.
                chunk = empty_chunk_frame()
                refetch = False
                print(f"[MLB] resume existing empty chunk {cursor}..{chunk_end}")
            else:
                try:
                    chunk = norm(pd.read_csv(path, low_memory=False))
                    print(f"[MLB] resume existing chunk {cursor}..{chunk_end}: {len(chunk)} rows")
                except pd.errors.EmptyDataError:
                    refetch = True
                    reason = "empty-data checkpoint"
                except Exception as exc:
                    raise RuntimeError(f"MLB checkpoint chunk unreadable: {path}: {exc}") from exc
                if not refetch and "game_type" not in chunk.columns:
                    refetch = True
                    reason = "checkpoint lacks official game_type"
        if refetch:
            quarantine = None
            if path.exists():
                quarantine = path.with_suffix(path.suffix + ".corrupt")
                if quarantine.exists():
                    raise RuntimeError(
                        f"MLB checkpoint remains corrupt and has an existing quarantine: {path}"
                    )
                path.replace(quarantine)
                print(f"[MLB] {reason}; quarantined {quarantine.name}")
            else:
                print(f"[MLB] fetch chunk {cursor}..{chunk_end}")
            chunk = norm(fetch_schedule(cursor, chunk_end))
            atomic_csv(chunk, path)
            print(f"[MLB] checkpointed chunk {path.name}: {len(chunk)} rows")
        frames.append(chunk)
        completed.append({
            "start": str(cursor),
            "end": str(chunk_end),
            "path": str(path),
            "rows": int(len(chunk)),
        })
        cursor = chunk_end + timedelta(days=1)
    if not frames:
        return pd.DataFrame(), completed
    return norm(pd.concat(frames, ignore_index=True, sort=False)), completed


def load_manifest():
    if not MANIFEST.exists():
        return {"schema_version": 1, "completed_ranges": []}
    try:
        return json.loads(MANIFEST.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"MLB acquisition manifest unreadable: {exc}") from exc


def save_manifest(payload):
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    tmp = MANIFEST.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(MANIFEST)


def enrich_missing_starters(df):
    changed = 0
    for i, row in df.iterrows():
        hs = str(row.get("home_starter", "") or "").strip()
        ass = str(row.get("away_starter", "") or "").strip()
        if hs and ass:
            continue
        gid = str(row.get("game_id", ""))
        if not gid or gid == "nan":
            continue
        try:
            feed = get_json(f"{API}/game/{gid}/feed/live")
            teams = feed.get("liveData", {}).get("boxscore", {}).get("teams", {})
            def find_starter(side):
                players = teams.get(side, {}).get("players", {})
                for p in players.values():
                    pit = p.get("stats", {}).get("pitching", {})
                    if pit.get("gamesStarted", 0) == 1:
                        return p.get("person", {}).get("fullName", "")
                for p in players.values():
                    if p.get("gameStatus", {}).get("isStartingPitcher"):
                        return p.get("person", {}).get("fullName", "")
                return ""
            h2, a2 = find_starter("home"), find_starter("away")
            if h2 and h2 != hs:
                df.at[i, "home_starter"] = h2
                changed += 1
            if a2 and a2 != ass:
                df.at[i, "away_starter"] = a2
                changed += 1
            df.at[i, "confirmed_starters"] = bool(
                df.at[i, "home_starter"] and df.at[i, "away_starter"]
            )
        except Exception as e:
            print(f"[MLB] starter skip game={gid}: {e}")
    return changed


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    CHUNK_DIR.mkdir(parents=True, exist_ok=True)

    existing = read_existing_cache()
    manifest = load_manifest()

    # IMPORTANT: no historical data is discarded. An empty cache triggers a
    # complete bootstrap from the configured historical start through today,
    # using resumable date chunks.
    if existing.empty:
        start = datetime(BOOTSTRAP_START_YEAR, 3, 1, tzinfo=timezone.utc).date()
        bootstrap = True
        fresh, ranges = acquire_chunked(start, TODAY)
        combined = norm(fresh)
    else:
        current = existing[existing["datetime"].dt.year == SEASON]
        if current.empty:
            start = datetime(SEASON, 3, 1, tzinfo=timezone.utc).date()
        else:
            start = max(
                datetime(SEASON, 3, 1, tzinfo=timezone.utc).date(),
                current["datetime"].max().date() - timedelta(days=CORRECTION_DAYS),
            )
        end = TODAY
        bootstrap = False
        if start > end:
            start = end
        fresh = norm(fetch_schedule(start, end))
        ranges = [{
            "start": str(start),
            "end": str(end),
            "path": "incremental-live",
            "rows": int(len(fresh)),
        }]
        combined = norm(pd.concat([existing, fresh], ignore_index=True, sort=False))

    mode = "FULL HISTORICAL BOOTSTRAP" if bootstrap else "INCREMENTAL DELTA"
    print(f"[MLB] mode={mode} cache={len(existing)} rows")
    print(f"[MLB] acquisition ranges={len(ranges)}")

    missing = (
        (combined["home_starter"].fillna("").astype(str).str.strip() == "")
        | (combined["away_starter"].fillna("").astype(str).str.strip() == "")
    )
    current_or_recent = combined["datetime"] >= pd.Timestamp(
        datetime.now(timezone.utc) - timedelta(days=CORRECTION_DAYS)
    )
    target = combined[missing & current_or_recent].copy()
    changed = enrich_missing_starters(target)
    if not target.empty:
        combined = combined.set_index("game_id")
        for _, r in target.iterrows():
            gid = str(r["game_id"])
            for c in ["home_starter", "away_starter", "confirmed_starters"]:
                combined.at[gid, c] = r[c]
        combined = combined.reset_index()

    atomic_csv(combined, CACHE)

    manifest.update({
        "schema_version": 1,
        "source": API,
        "bootstrap_start_year": BOOTSTRAP_START_YEAR,
        "chunk_days": CHUNK_DAYS,
        "last_mode": mode,
        "last_run_utc": datetime.now(timezone.utc).isoformat(),
        "last_start": str(start),
        "last_end": str(TODAY),
        "completed_ranges_tail": ranges[-50:],
        "total_rows": int(len(combined)),
    })
    save_manifest(manifest)

    print(f"[MLB] starter_repairs={changed} total={len(combined)}")
    if len(combined) < 100:
        # Do not publish a misleading readiness contract for a partial or
        # unexpectedly small historical acquisition.
        raise RuntimeError("MLB cache unexpectedly small; full-scope acquisition was not completed")

    # A successful full-scope bootstrap must emit the readiness contract that
    # the production gate consumes. This is distinct from the resumable
    # manifest: readiness is only true after the complete requested range has
    # been processed without an acquisition exception and the cache-size gate
    # has passed.
    status = {
        "schema_version": 1,
        "complete": True,
        "aggregate_games": int(len(combined)),
        "bootstrap_start_year": BOOTSTRAP_START_YEAR,
        "last_start": str(start),
        "last_end": str(TODAY),
        "source": API,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    save_status = CHECKPOINTS / "mlb_collection_status.json"
    tmp_status = save_status.with_suffix(".json.tmp")
    tmp_status.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp_status.replace(save_status)


if __name__ == "__main__":
    main()
