#!/usr/bin/env python3
"""Data collection pipeline for MLB/NPB"""
import logging
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from core import storage, config

def collect_mlb():
    """Collect MLB game data from StatsAPI"""
    import urllib.request, json
    
    cfg = config.load()
    data_dir = storage.path("raw")
    
    for season in range(cfg.get("mlb", {}).get("start_season", 2024), cfg.get("mlb", {}).get("end_season", 2026) + 1):
        url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&startDate={season}-01-01&endDate={season}-12-31&gameType=R"
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                data = json.load(r)
            rows = []
            for d in data.get("dates", []):
                for g in d.get("games", []):
                    hs = g.get("teams", {}).get("home", {}).get("score")
                    aws = g.get("teams", {}).get("away", {}).get("score")
                    status = g.get("status", {}).get("detailedState")
                    if status == "Final" and hs is not None and aws is not None:
                        rows.append({
                            "date": d.get("date"),
                            "home_team": g.get("teams", {}).get("home", {}).get("team", {}).get("name"),
                            "away_team": g.get("teams", {}).get("away", {}).get("team", {}).get("name"),
                            "home_score": hs,
                            "away_score": aws,
                            "league": "MLB"
                        })
            df = pd.DataFrame(rows)
            if not df.empty:
                storage.write(df, "raw", f"mlb_games_{season}", partition=str(season))
                print(f"Season {season}: {len(df)} games")
        except Exception as e:
            print(f"Season {season} failed: {e}")

if __name__ == "__main__":
    collect_mlb()
