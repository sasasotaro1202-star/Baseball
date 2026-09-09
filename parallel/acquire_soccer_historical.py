"""Historical soccer data acquisition from football-data.co.uk (free, public).

This script is designed to run inside GitHub Actions (which has real
internet access) -- NOT in a sandboxed/offline environment. It downloads
~15 seasons of match + odds data for each target league and normalizes
column names to the schema expected by parallel/data_loader.py.

Coverage note: football-data.co.uk provides categories 1 (match basic
data) and 2 (market odds) from prediction_model_specification.md. It does
NOT provide Understat xG (category 3) or SofaScore player data (category
8-11) -- those require separate acquisition and are not yet automated.

Usage (inside CI):
    python -m parallel.acquire_soccer_historical --years 15
"""
import argparse
import io
import os
import sys
import time

import pandas as pd
import requests

# football-data.co.uk division codes for leagues covered by this free source.
# Note: J-League, UCL/UEL, DFB-Pokal, and Club Friendlies are NOT available
# from this source and require separate acquisition (see docs/PARALLELIZATION.md).
LEAGUE_CODES = {
    "premier_league": "E0",
    "bundesliga": "D1",
    "serie_a": "I1",
    "la_liga": "SP1",
    "ligue_1": "F1",
    "eredivisie": "N1",
}

BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/{code}.csv"

RENAME_MAP = {
    "Date": "date", "HomeTeam": "home_team", "AwayTeam": "away_team",
    "FTHG": "home_score", "FTAG": "away_score", "FTR": "FTR",
    "HS": "shots_home", "AS": "shots_away", "HST": "sot_home", "AST": "sot_away",
    "HC": "corners_home", "AC": "corners_away",
}
KEEP_COLS = list(RENAME_MAP.keys())


def season_code(end_year: int) -> str:
    """end_year=2026 -> '2526' (2025/26 season)."""
    start_year = end_year - 1
    return f"{start_year % 100:02d}{end_year % 100:02d}"


def fetch_season_csv(league_code: str, season: str, timeout: int = 15):
    url = BASE_URL.format(season=season, code=league_code)
    try:
        resp = requests.get(url, timeout=timeout)
        if resp.status_code != 200 or len(resp.content) < 100:
            return None
        df = pd.read_csv(io.StringIO(resp.content.decode("utf-8", errors="ignore")))
    except Exception as e:
        print(f"[warn] failed to fetch {url}: {e}")
        return None
    cols = [c for c in KEEP_COLS if c in df.columns]
    if "Date" not in cols or "HomeTeam" not in cols:
        return None
    df = df[cols].rename(columns=RENAME_MAP)
    try:
        df["date"] = pd.to_datetime(df["date"], dayfirst=True).dt.strftime("%Y-%m-%d")
    except Exception:
        return None
    return df


def acquire(years: int, out_dir: str = "data/soccer/historical", sleep_sec: float = 1.0) -> dict:
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
