#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Free official-source acquisition for Asian Games baseball.

This adapter intentionally targets official Japan Baseball pages because they
publish tournament schedules/results in a stable tabular format and explicitly
state that displayed times are JST. It writes only completed games with final
scores; future fixtures are never turned into historical training rows.

The resulting CSV follows the shared INTERNATIONAL schema consumed by
baseball_backtest.py.
"""
from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "international_games.csv"
JST = ZoneInfo("Asia/Tokyo")
TIMEOUT = 25

ASIAN_GAMES = {
    2014: "https://www.japan-baseball.jp/en/team/amateur/2014/asiangames/overview.html",
    2018: "https://www.japan-baseball.jp/en/team/amateur/2018/asiangames/overview.html",
    2022: "https://www.japan-baseball.jp/en/team/amateur/2022/asiangames/overview.html",
    2023: "https://www.japan-baseball.jp/en/team/amateur/2023/asiangames/overview.html",
    2026: "https://www.japan-baseball.jp/en/team/amateur/2026/asiangames/overview.html",
}


def fetch(url: str) -> str:
    last = None
    for attempt in range(4):
        try:
            r = requests.get(
                url,
                timeout=TIMEOUT,
                headers={"User-Agent": "Baseball-International-Research/1.0"},
            )
            r.raise_for_status()
            return r.text
        except Exception as exc:
            last = exc
            if attempt < 3:
                import time
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"request failed: {url}: {last}")


def parse_matchup(value: object) -> tuple[str, int, int, str] | None:
    text = " ".join(str(value or "").replace("\xa0", " ").split())
    m = re.match(r"^(.+?)\s+(\d+)\s*-\s*(\d+)\s+(.+?)$", text)
    if not m:
        return None
    home = m.group(1).strip()
    away = m.group(4).strip()
    if not home or not away:
        return None
    return home, int(m.group(2)), int(m.group(3)), away


def parse_year(year: int, url: str) -> list[dict]:
    html = fetch(url)
    tables = pd.read_html(html)
    rows: list[dict] = []
    for table in tables:
        cols = {str(c).strip(): c for c in table.columns}
        date_col = next((cols[c] for c in cols if "Date and time" in c), None)
        match_col = next((cols[c] for c in cols if "Home - Visitor" in c), None)
        if date_col is None or match_col is None:
            continue
        for _, row in table.iterrows():
            date_text = " ".join(str(row.get(date_col, "")).split())
            matchup = parse_matchup(row.get(match_col, ""))
            if matchup is None:
                continue
            m = re.search(r"(\d{1,2}/\d{1,2}/\d{4})", date_text)
            if not m:
                continue
            d = datetime.strptime(m.group(1), "%m/%d/%Y").replace(tzinfo=JST)
            home, hs, aas, away = matchup
            key = f"asian_games:{year}:{d.isoformat()}:{home}:{away}:{hs}:{aas}"
            gid = "intl_ag_" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:20]
            rows.append({
                "league": "INTERNATIONAL",
                "game_id": gid,
                "datetime": d.astimezone(timezone.utc).isoformat(),
                "home": home,
                "away": away,
                "home_score": hs,
                "away_score": aas,
                "competition": f"{year} Asian Games",
                "game_type": "Asian Games",
                "series_description": "Asian Games Baseball",
                "source_url": url,
            })
    return rows


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    failures = []
    for year, url in ASIAN_GAMES.items():
        try:
            rows = parse_year(year, url)
            all_rows.extend(rows)
            print(f"[ASIAN-GAMES] {year}: {len(rows)} completed games")
        except Exception as exc:
            failures.append({"year": year, "url": url, "error": str(exc)})
            print(f"[ASIAN-GAMES] {year}: DEFERRED: {exc}")

    if not all_rows:
        raise RuntimeError("No Asian Games completed games were acquired.")

    df = pd.DataFrame(all_rows)
    df = df.sort_values(["datetime", "game_id"]).drop_duplicates("game_id", keep="last")
    tmp = OUT.with_suffix(".tmp.csv")
    df.to_csv(tmp, index=False)
    tmp.replace(OUT)

    print(f"[ASIAN-GAMES] wrote {len(df)} games -> {OUT}")
    if failures:
        print(f"[ASIAN-GAMES] deferred sources={len(failures)}; no synthetic rows created")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
