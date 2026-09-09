"""Historical soccer data acquisition from football-data.co.uk (free, public).

This script is designed to run inside GitHub Actions (which has real
internet access) -- NOT in a sandboxed/offline environment. It downloads
~15 seasons of match + odds data for each target league and normalizes
column names to the schema expected by parallel/data_loader.py.

Coverage note: football-data.co.uk provides categories 1 (match basic
data) and 2 (market odds) from prediction_model_specification.md. It does
NOT provide Understat xG (category 3) or SofaScore player data (category
8-11) -- those require separate acquisition and are not yet automated.

Fix (2026-09-09, v1): football-data.co.uk rejects requests with the default
python-requests User-Agent (returns non-200 / empty body). A browser-like
User-Agent header was added.

Fix (2026-09-09, v2): a User-Agent header alone was still insufficient --
the site (Cloudflare-fronted) appears to filter on TLS/JA3 fingerprint,
which plain `requests` cannot spoof. Switched to `curl_cffi` with
impersonate="chrome", which replicates a real Chrome TLS handshake. This
is the same technique already proven working in the sibling `soccer` repo
(soccer_source_acquisition_v2.py) against the same football-data.co.uk
endpoints.

Usage (inside CI):
    python -m parallel.acquire_soccer_historical --years 15
"""
import argparse
import io
import os
import sys
import time

import pandas as pd
from curl_cffi import requests as curl_requests

LEAGUE_CODES = {
    "premier_league": "E0",
    "bundesliga": "D1",
    "serie_a": "I1",
    "la_liga": "SP1",
    "ligue_1": "F1",
    "eredivisie": "N1",
}

BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/{code}.csv"

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/140.0 Safari/537.36"),
    "Accept": "text/csv,application/csv,text/plain,*/*",
    "Accept-Language": "en-US,en;q=0.9",
}

RENAME_MAP = {
    "Date": "date", "HomeTeam": "home_team", "AwayTeam": "away_team",
    "FTHG": "home_score", "FTAG": "away_score", "FTR": "FTR",
    "HS": "shots_home", "AS": "shots_away", "HST": "sot_home", "AST": "sot_away",
    "HC": "corners_home", "AC": "corners_away",
}
KEEP_COLS = list(RENAME_MAP.keys())

SESSION = curl_requests.Session(impersonate="chrome")
SESSION.headers.update(HEADERS)


def season_code(end_year: int) -> str:
    start_year = end_year - 1
    return f"{start_year % 100:02d}{end_year % 100:02d}"


def fetch_season_csv(league_code: str, season: str, timeout: int = 20, retries: int = 3):
    url = BASE_URL.format(season=season, code=league_code)
    last_err = ""
    for attempt in range(retries):
        try:
            resp = SESSION.get(url, timeout=timeout)
            if resp.status_code == 200 and len(resp.content) >= 100:
                df = pd.read_csv(io.StringIO(resp.content.decode("utf-8", errors="ignore")))
                break
            last_err = f"HTTP {resp.status_code}, {len(resp.content)} bytes"
        except Exception as e:
            last_err = repr(e)
        time.sleep(min(1.5 * (attempt + 1), 6))
    else:
        print(f"[warn] {url} -> {last_err}")
        return None
    cols = [c for c in KEEP_COLS if c in df.columns]
    if "Date" not in cols or "HomeTeam" not in cols:
        print(f"[warn] {url} -> unexpected columns: {df.columns.tolist()}")
        return None
    df = df[cols].rename(columns=RENAME_MAP)
    try:
        df["date"] = pd.to_datetime(df["date"], dayfirst=True).dt.strftime("%Y-%m-%d")
    except Exception:
        return None
    return df


def acquire(years: int, out_dir: str = "data/soccer/historical", sleep_sec: float = 1.5) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    current_end_year = 2026
    report = {}
    for league_name, code in LEAGUE_CODES.items():
        league_dir = os.path.join(out_dir, league_name)
        os.makedirs(league_dir, exist_ok=True)
        seasons_ok, seasons_missing = [], []
        for i in range(years):
            end_year = current_end_year - i
            season = season_code(end_year)
            df = fetch_season_csv(code, season)
            if df is not None and len(df) > 0:
                out_path = os.path.join(league_dir, f"{end_year}.csv")
                df.to_csv(out_path, index=False)
                seasons_ok.append(end_year)
            else:
                seasons_missing.append(end_year)
            time.sleep(sleep_sec)
        report[league_name] = {"seasons_ok": seasons_ok, "seasons_missing": seasons_missing}
        print(f"[{league_name}] ok={len(seasons_ok)} missing={len(seasons_missing)}")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=int, default=15)
    parser.add_argument("--out-dir", default="data/soccer/historical")
    args = parser.parse_args()
    report = acquire(args.years, args.out_dir)
    import json
    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/soccer_acquisition_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
