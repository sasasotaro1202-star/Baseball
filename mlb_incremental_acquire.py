#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Full-scope, incremental MLB source acquisition.

The dataset scope is NOT reduced: the first bootstrap acquires the full
historical window, then subsequent runs refresh only the delta while retaining
all historical rows locally.
"""
from __future__ import annotations

import os
from pathlib import Path
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

from mlb_game_type import classify_mlb_game

API = "https://statsapi.mlb.com/api/v1"
DATA = Path(os.getenv("BASEBALL_DATA_DIR", "data"))
CACHE = DATA / "mlb_games.csv"
TIMEOUT = int(os.getenv("MLB_INCREMENTAL_TIMEOUT", "20"))
CORRECTION_DAYS = int(os.getenv("MLB_CORRECTION_DAYS", "7"))
BOOTSTRAP_START_YEAR = int(os.getenv("MLB_BOOTSTRAP_START_YEAR", "2020"))
TODAY = datetime.now(timezone.utc).date()
SEASON = TODAY.year

S = requests.Session()
S.headers.update({"User-Agent": "BaseballIncrementalAcquisition/1.1", "Accept": "application/json"})

# Conservative minimum final-game counts used only to detect obviously incomplete
# historical seasons. These are data-quality gates, not assumed exact schedules.
MIN_COMPLETED_GAMES_BY_YEAR = {
    2020: 800,
    2021: 2000,
    2022: 2000,
    2023: 2000,
    2024: 2000,
    2025: 2000,
}


def history_coverage(df: pd.DataFrame, start_year: int, end_year: int) -> dict[int, int]:
    if df.empty or "datetime" not in df.columns:
        return {year: 0 for year in range(start_year, end_year + 1)}
    years = pd.to_datetime(df["datetime"], errors="coerce", utc=True).dt.year
    return {year: int((years == year).sum()) for year in range(start_year, end_year + 1)}


def incomplete_historical_years(df: pd.DataFrame, start_year: int, end_year: int) -> list[int]:
    coverage = history_coverage(df, start_year, end_year)
    return [
        year for year, minimum in MIN_COMPLETED_GAMES_BY_YEAR.items()
        if start_year <= year <= end_year and coverage.get(year, 0) < minimum
    ]


def validate_historical_coverage(df: pd.DataFrame, start_year: int, end_year: int, allow_current_partial: bool = True) -> None:
    coverage = history_coverage(df, start_year, end_year)
    missing = []
    for year, minimum in MIN_COMPLETED_GAMES_BY_YEAR.items():
        if not (start_year <= year <= end_year):
            continue
        if coverage.get(year, 0) < minimum:
            missing.append({"year": year, "rows": coverage.get(year, 0), "minimum": minimum})
    if missing:
        raise RuntimeError(f"MLB historical coverage incomplete: {missing}")
    print(f"[MLB] historical coverage PASS: {coverage}")



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
            import time
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
    for col, default in [("game_type_code", ""), ("series_description", ""), ("game_type", "UNKNOWN")]:
        if col not in df.columns:
            df[col] = default
    classified = df.apply(
        lambda r: classify_mlb_game(
            r.get("game_type_code", ""),
            r.get("series_description", ""),
            r.get("game_type", ""),
        ),
        axis=1,
    )
    df["mlb_game_category"] = classified.map(lambda x: x["category"])
    df["mlb_type_confidence"] = classified.map(lambda x: x["confidence"])
    df["mlb_training_default"] = classified.map(lambda x: bool(x["training_default"]))
    df["mlb_evaluation_default"] = classified.map(lambda x: bool(x["evaluation_default"]))
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
                "game_type_code": str(game.get("gameType") or ""),
                "series_description": str(game.get("seriesDescription") or ""),
                "game_type": str(game.get("seriesDescription") or game.get("gameType") or "UNKNOWN"),
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
            print(f"[MLB] starter skip game={gid}: {e}")
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

    # IMPORTANT: no historical data is discarded. An empty cache triggers a
    # complete bootstrap from the configured historical start through today.
    if existing.empty:
        start = datetime(BOOTSTRAP_START_YEAR, 3, 1, tzinfo=timezone.utc).date()
        bootstrap = True
    else:
        current = existing[existing["datetime"].dt.year == SEASON]
        if current.empty:
            start = datetime(SEASON, 3, 1, tzinfo=timezone.utc).date()
        else:
            start = max(datetime(SEASON, 3, 1, tzinfo=timezone.utc).date(), current["datetime"].max().date() - timedelta(days=CORRECTION_DAYS))
        bootstrap = False

    end = TODAY
    if start > end:
        start = end

    # A non-empty but truncated cache must never silently switch to incremental
    # mode. Re-acquire every obviously incomplete completed season before using
    # the dataset for OOS/modeling. Current season remains a delta refresh.
    historical_years = [y for y in range(BOOTSTRAP_START_YEAR, SEASON) if y >= BOOTSTRAP_START_YEAR]
    missing_years = incomplete_historical_years(existing, BOOTSTRAP_START_YEAR, max(SEASON - 1, BOOTSTRAP_START_YEAR))
    mode = "FULL HISTORICAL RECOVERY" if missing_years else ("FULL HISTORICAL BOOTSTRAP" if bootstrap else "INCREMENTAL DELTA")
    print(f"[MLB] mode={mode} cache={len(existing)} rows; missing_years={missing_years}")

    parts = []
    for year in missing_years:
        year_start = datetime(year, 3, 1, tzinfo=timezone.utc).date()
        year_end = datetime(year, 11, 30, tzinfo=timezone.utc).date()
        print(f"[MLB] recovering historical season {year}: {year_start}..{year_end}")
        parts.append(fetch_schedule(year_start, year_end))

    # If this is a true empty bootstrap, fetch every historical year in range.
    if bootstrap:
        parts = [fetch_schedule(
            datetime(year, 3, 1, tzinfo=timezone.utc).date(),
            datetime(year, 11, 30, tzinfo=timezone.utc).date(),
        ) for year in range(BOOTSTRAP_START_YEAR, SEASON)] + parts

    # Always refresh the current season from the last correction window onward.
    current_start = start
    parts.append(fetch_schedule(current_start, end))
    fresh = norm(pd.concat(parts, ignore_index=True, sort=False)) if parts else pd.DataFrame()
    combined = norm(pd.concat([existing, fresh], ignore_index=True, sort=False)) if not existing.empty else fresh

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
                combined.at[gid, c] = r[c]
        combined = combined.reset_index()

    combined = norm(combined)
    validate_historical_coverage(combined, BOOTSTRAP_START_YEAR, SEASON - 1)
    combined.to_csv(CACHE, index=False)
    print(f"[MLB] fresh={len(fresh)} starter_repairs={changed} total={len(combined)}")
    if len(combined) < 100:
        raise RuntimeError("MLB cache unexpectedly small; full-scope acquisition was not completed")


if __name__ == "__main__":
    main()
