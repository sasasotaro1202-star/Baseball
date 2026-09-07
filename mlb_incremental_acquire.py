#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Incremental MLB source acquisition.

Keeps data/mlb_games.csv as the durable cache and only queries the MLB Stats API
for the uncached portion of the current season plus a small correction window.
Starter enrichment is performed only for games whose starter fields are missing.
"""
from __future__ import annotations

import os
from pathlib import Path
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

API = "https://statsapi.mlb.com/api/v1"
DATA = Path(os.getenv("BASEBALL_DATA_DIR", "data"))
CACHE = DATA / "mlb_games.csv"
TIMEOUT = int(os.getenv("MLB_INCREMENTAL_TIMEOUT", "20"))
CORRECTION_DAYS = int(os.getenv("MLB_CORRECTION_DAYS", "7"))
TODAY = datetime.now(timezone.utc).date()
SEASON = TODAY.year

S = requests.Session()
S.headers.update({"User-Agent": "BaseballIncrementalAcquisition/1.0", "Accept": "application/json"})


def get_json(url, params=None):
    r = S.get(url, params=params, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


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
                "confirmed_starters": bool((home.get("probablePitcher") or {}).get("fullName") and (away.get("probablePitcher") or {}).get("fullName")),
            })
    return pd.DataFrame(rows)


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
                df.at[i, "home_starter"] = h2; changed += 1
            if a2 and a2 != ass:
                df.at[i, "away_starter"] = a2; changed += 1
            df.at[i, "confirmed_starters"] = bool(df.at[i, "home_starter"] and df.at[i, "away_starter"])
        except Exception as e:
            print(f"[MLB INCR] starter skip game={gid}: {e}")
    return changed


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    if CACHE.exists():
        try:
            existing = norm(pd.read_csv(CACHE, low_memory=False))
        except Exception:
            existing = pd.DataFrame()
    else:
        existing = pd.DataFrame()

    if existing.empty:
        start = datetime(SEASON, 3, 1, tzinfo=timezone.utc).date()
    else:
        current = existing[existing["datetime"].dt.year == SEASON]
        if current.empty:
            start = datetime(SEASON, 3, 1, tzinfo=timezone.utc).date()
        else:
            start = max(datetime(SEASON, 3, 1, tzinfo=timezone.utc).date(), (current["datetime"].max().date() - timedelta(days=CORRECTION_DAYS)))

    # Never query beyond today; the correction window only re-reads recent completed games.
    end = TODAY
    if start > end:
        start = end

    print(f"[MLB INCR] cache={len(existing)} rows; querying {start}..{end}")
    fresh = fetch_schedule(start, end)
    fresh = norm(fresh)
    combined = norm(pd.concat([existing, fresh], ignore_index=True, sort=False)) if not existing.empty else fresh
    combined = combined.sort_values(["datetime", "game_id"]).drop_duplicates("game_id", keep="last").reset_index(drop=True)

    # Only repair missing starters. This replaces the previous O(all-history) feed sweep.
    missing = ((combined["home_starter"].fillna("").astype(str).str.strip() == "") |
               (combined["away_starter"].fillna("").astype(str).str.strip() == ""))
    current_or_recent = combined["datetime"] >= pd.Timestamp(datetime.now(timezone.utc) - timedelta(days=CORRECTION_DAYS))
    target = combined[missing & current_or_recent].copy()
    changed = enrich_missing_starters(target)
    if not target.empty:
        combined = combined.set_index("game_id")
        for _, r in target.iterrows():
            gid = str(r["game_id"])
            for c in ["home_starter", "away_starter", "confirmed_starters"]:
                if c in r:
                    combined.at[gid, c] = r[c]
        combined = combined.reset_index()

    combined.to_csv(CACHE, index=False)
    print(f"[MLB INCR] fresh={len(fresh)} starter_repairs={changed} total={len(combined)}")


if __name__ == "__main__":
    main()
