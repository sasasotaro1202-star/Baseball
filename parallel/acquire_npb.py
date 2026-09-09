"""NPB (Nippon Professional Baseball) historical data acquisition.

Source: armstjc/Nippon-Baseball-Data-Repository (MIT licensed, public GitHub repo)
https://github.com/armstjc/Nippon-Baseball-Data-Repository

Coverage: 2018-2025 NPB schedules (8 seasons). This is NOT 15 years, but is
the best free structured source found for NPB to date. If a longer-history
NPB source is found later, extend SEASONS below.

Honesty note: column names in the source CSVs were not verified before
writing this script (raw.githubusercontent.com could not be previewed from
the assistant's tools). This script auto-detects likely column names for
date/home/away/score and logs a warning + saves the raw file untouched if
detection fails, so no data is silently mislabeled.
"""
import argparse
import io
import os
import sys

import pandas as pd
import requests

REPO_RAW_BASE = "https://raw.githubusercontent.com/armstjc/Nippon-Baseball-Data-Repository/main/schedules"
SEASONS = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]

DATE_CANDIDATES = ["date", "Date", "game_date", "GameDate"]
HOME_CANDIDATES = ["home_team", "HomeTeam", "home", "Home", "home_team_name"]
AWAY_CANDIDATES = ["away_team", "AwayTeam", "away", "Away", "away_team_name"]
HOME_SCORE_CANDIDATES = ["home_score", "HomeScore", "home_runs", "HomeRuns"]
AWAY_SCORE_CANDIDATES = ["away_score", "AwayScore", "away_runs", "AwayRuns"]


def _find_col(df_columns, candidates):
    for c in candidates:
        if c in df_columns:
            return c
    return None


def fetch_season(year: int, timeout: int = 20):
    url = f"{REPO_RAW_BASE}/{year}_npb_schedule.csv"
    resp = requests.get(url, timeout=timeout)
    if resp.status_code != 200:
        print(f"[warn] {year}: HTTP {resp.status_code}")
        return None
    df = pd.read_csv(io.StringIO(resp.content.decode("utf-8", errors="ignore")))
    return df


def normalize(df: pd.DataFrame, year: int):
    cols = df.columns.tolist()
    date_col = _find_col(cols, DATE_CANDIDATES)
    home_col = _find_col(cols, HOME_CANDIDATES)
    away_col = _find_col(cols, AWAY_CANDIDATES)
    hs_col = _find_col(cols, HOME_SCORE_CANDIDATES)
    as_col = _find_col(cols, AWAY_SCORE_CANDIDATES)

    detected = {"date": date_col, "home_team": home_col, "away_team": away_col,
                "home_score": hs_col, "away_score": as_col}
    missing = [k for k, v in detected.items() if v is None]
    if missing:
        print(f"[warn] {year}: could not auto-detect columns {missing}. "
              f"Raw columns were: {cols}. Saving RAW (unnormalized) file instead.")
        return df, False

    out = df.rename(columns={date_col: "date", home_col: "home_team", away_col: "away_team",
                              hs_col: "home_score", as_col: "away_score"})
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out["league"] = "NPB"
    return out, True


def acquire(out_dir: str = "data/npb"):
    os.makedirs(out_dir, exist_ok=True)
    report = {}
    for year in SEASONS:
        df = fetch_season(year)
        if df is None:
            report[year] = "fetch_failed"
            continue
        normalized, ok = normalize(df, year)
        suffix = "" if ok else "_RAW_UNNORMALIZED"
        out_path = os.path.join(out_dir, f"npb_games_{year}{suffix}.csv")
        normalized.to_csv(out_path, index=False)
        report[year] = "ok" if ok else "saved_raw_needs_manual_mapping"
        print(f"[{year}] status={report[year]} rows={len(normalized)}")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data/npb")
    args = parser.parse_args()
    report = acquire(args.out_dir)
    import json
    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/npb_acquisition_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
