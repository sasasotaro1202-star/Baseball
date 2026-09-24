#!/usr/bin/env python3
"""Free, official NPB settlement for the Matchday forward ledger.

Only completed historical schedule rows are used to attach outcomes. Prediction
timestamps and probabilities are never changed. This closes the forward loop:
predict -> persist snapshot -> wait for official result -> settle -> score.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

RESULTS = Path("results")
LEDGER = RESULTS / "matchday_baseline.csv"
SETTLEMENT = RESULTS / "matchday_settlement.json"
NPB = "https://npb.jp"
JST = ZoneInfo("Asia/Tokyo")
TIMEOUT = 25

ALIASES = {
    "巨人":"読売ジャイアンツ","読売":"読売ジャイアンツ","読売ジャイアンツ":"読売ジャイアンツ",
    "阪神":"阪神タイガース","中日":"中日ドラゴンズ","広島":"広島東洋カープ",
    "ヤクルト":"東京ヤクルトスワローズ","DeNA":"横浜DeNAベイスターズ",
    "横浜":"横浜DeNAベイスターズ","ソフトバンク":"福岡ソフトバンクホークス",
    "西武":"埼玉西武ライオンズ","日本ハム":"北海道日本ハムファイターズ",
    "日ハム":"北海道日本ハムファイターズ","ロッテ":"千葉ロッテマリーンズ",
    "楽天":"東北楽天ゴールデンイーグルス","オリックス":"オリックス・バファローズ",
}
TEAMS = set(ALIASES.values())


def norm_team(x: str) -> str:
    return ALIASES.get(str(x or "").strip(), str(x or "").strip())


def get(url: str):
    r = requests.get(url, timeout=TIMEOUT, headers={"User-Agent":"Baseball-Matchday-Settlement/1.0"})
    r.raise_for_status()
    return r.content


def parse_schedule_month(year: int, month: int):
    url = f"{NPB}/games/{year}/schedule_{month:02d}_detail.html"
    tables = pd.read_html(get(url))
    rows = []
    for table in tables:
        if "月日" not in table.columns or "対戦カード" not in table.columns:
            continue
        for _, rec in table.iterrows():
            date_text = re.sub(r"\s+", " ", str(rec.get("月日",""))).strip()
            mdate = re.match(r"^(\d{1,2})/(\d{1,2})", date_text)
            if not mdate:
                continue
            day = int(mdate.group(2))
            card = re.sub(r"\s+", " ", str(rec.get("対戦カード",""))).strip()
            m = re.match(r"^(.+?)\s+(\d+)\s*-\s*(\d+)\s+(.+?)$", card)
            if not m:
                continue
            home = norm_team(m.group(1))
            away = norm_team(m.group(4))
            if home not in TEAMS or away not in TEAMS:
                continue
            hs, aw = int(m.group(2)), int(m.group(3))
            rows.append({
                "date": f"{year:04d}-{month:02d}-{day:02d}",
                "home": home,
                "away": away,
                "home_score": hs,
                "away_score": aw,
            })
    return rows


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    if not LEDGER.exists() or LEDGER.stat().st_size == 0:
        payload = {
            "schema_version": 1,
            "status": "DEFERRED",
            "settled_rows": 0,
            "reason": "forward Matchday ledger is missing",
        }
        SETTLEMENT.write_text(json.dumps(payload, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
        print(json.dumps(payload, ensure_ascii=False))
        return 0

    df = pd.read_csv(LEDGER)
    if "actual" not in df.columns:
        df["actual"] = float("nan")
    dt = pd.to_datetime(df.get("datetime"), errors="coerce", utc=True)
    df["_target_date"] = dt.dt.tz_convert(JST).dt.strftime("%Y-%m-%d")
    now_jst = datetime.now(JST)

    needed_dates = sorted(
        set(df.loc[df["actual"].isna(), "_target_date"].dropna().astype(str))
    )
    schedule_rows = []
    for d in needed_dates:
        try:
            y, m, _ = [int(x) for x in d.split("-")]
            schedule_rows.extend(parse_schedule_month(y, m))
        except Exception:
            continue

    lookup = {
        (x["date"], x["home"], x["away"]): x
        for x in schedule_rows
    }
    settled = 0
    for idx, row in df.iterrows():
        if pd.notna(row.get("actual")):
            continue
        target = str(row.get("_target_date") or "")
        if not target or target > now_jst.strftime("%Y-%m-%d"):
            continue
        # Same-day games are held until a conservative 5-hour post-start buffer.
        # Older dates can settle immediately.
        game_dt = pd.to_datetime(row.get("datetime"), errors="coerce", utc=True)
        if pd.notna(game_dt) and target == now_jst.strftime("%Y-%m-%d"):
            elapsed = (pd.Timestamp.now(tz="UTC") - game_dt).total_seconds()
            if elapsed < 5 * 3600:
                continue
        home, away = norm_team(row.get("home")), norm_team(row.get("away"))
        result = lookup.get((target, home, away))
        if not result:
            continue
        hs, aw = result["home_score"], result["away_score"]
        if hs > aw:
            actual = 0
        elif hs == aw:
            actual = 1
        else:
            actual = 2
        df.at[idx, "actual"] = int(actual)
        df.at[idx, "actual_home_score"] = hs
        df.at[idx, "actual_away_score"] = aw
        settled += 1

    df.drop(columns=["_target_date"], inplace=True, errors="ignore")
    df.to_csv(LEDGER, index=False)

    payload = {
        "schema_version": 1,
        "status": "PASS",
        "settled_rows": int(settled),
        "settled_total": int(pd.to_numeric(df["actual"], errors="coerce").notna().sum()),
        "source": "NPB.jp official schedule/result",
        "same_day_buffer_hours": 5,
    }
    SETTLEMENT.write_text(json.dumps(payload, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
