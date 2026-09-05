#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Add an official NPB.jp validation layer to the NPB collector.

The granular SPAIA feed remains the efficient source for per-game enrichment,
but a completed schedule/result row is admitted only after its date, teams,
and (when published) score are corroborated by the official NPB.jp monthly
schedule-detail page. Missing/unparseable official validation is treated as
missing data, never as a reason to guess.
"""
from __future__ import annotations

import io
import re
from pathlib import Path

P = Path("npb_multi_source.py")
s = P.read_text(encoding="utf-8")
MARKER = "# OFFICIAL_NPB_SCHEDULE_VALIDATION_V1"
if MARKER in s:
    print("[OFFICIAL PATCH] V1 already applied")
    raise SystemExit(0)

anchor = "def fetch_games(year):\n"
if anchor not in s:
    raise RuntimeError("fetch_games anchor not found")

insert = r'''
# OFFICIAL_NPB_SCHEDULE_VALIDATION_V1
_OFFICIAL_SCHEDULE_CACHE = {}
_OFFICIAL_TEAM_SHORT = {
    "読売ジャイアンツ": "巨人", "阪神タイガース": "阪神", "中日ドラゴンズ": "中日",
    "広島東洋カープ": "広島", "東京ヤクルトスワローズ": "ヤクルト", "横浜DeNAベイスターズ": "DeNA",
    "福岡ソフトバンクホークス": "ソフトバンク", "埼玉西武ライオンズ": "西武", "北海道日本ハムファイターズ": "日本ハム",
    "千葉ロッテマリーンズ": "ロッテ", "東北楽天ゴールデンイーグルス": "楽天", "オリックス・バファローズ": "オリックス",
}


def _official_schedule_rows(year):
    """Return canonical (date, away, home, away_score, home_score) rows from NPB.jp."""
    if year in _OFFICIAL_SCHEDULE_CACHE:
        return _OFFICIAL_SCHEDULE_CACHE[year]
    rows = []
    for month in range(3, 12):
        if near_deadline():
            break
        url = f"{NPB}/games/{year}/schedule_{month:02d}_detail.html"
        try:
            r = requests.get(url, timeout=TIMEOUT, headers={"User-Agent": "Mozilla/5.0 baseball-backtest"})
            r.raise_for_status()
            tables = pd.read_html(io.StringIO(r.text))
        except Exception as exc:
            print(f"[OFFICIAL SCHEDULE SKIP] year={year} month={month}: {exc}")
            continue
        for table in tables:
            if "月日" not in table.columns or "対戦カード" not in table.columns:
                continue
            for _, rec in table.iterrows():
                date_text = str(rec.get("月日", ""))
                m = re.search(r"(\d{1,2})/(\d{1,2})", date_text)
                if not m:
                    continue
                mm, dd = int(m.group(1)), int(m.group(2))
                card = str(rec.get("対戦カード", ""))
                # read_html may concatenate multiple cells with whitespace; split
                # on line breaks and also handle the common single-card form.
                cards = [x.strip() for x in re.split(r"\n+", card) if x.strip()]
                if not cards:
                    cards = [card.strip()]
                for item in cards:
                    item = re.sub(r"\s+", " ", item).strip()
                    item = re.sub(r"\([^)]*\)", "", item).strip()
                    score = re.search(r"(.+?)\s+(\d+)\s*-\s*(\d+)\s+(.+)$", item)
                    if not score:
                        continue
                    left, away_score, home_score, right = score.groups()
                    away = official_name(left.strip())
                    home = official_name(right.strip())
                    # The official page displays away - home. Keep that explicit.
                    if away not in ALIASES.values() or home not in ALIASES.values():
                        continue
                    rows.append((f"{year:04d}-{mm:02d}-{dd:02d}", away, home, int(away_score), int(home_score)))
    _OFFICIAL_SCHEDULE_CACHE[year] = rows
    return rows


def _official_validate_games(year, games):
    """Filter SPAIA schedule rows through NPB.jp canonical identity/results."""
    if games.empty:
        return games
    official = _official_schedule_rows(year)
    if not official:
        raise RuntimeError(f"official NPB schedule validation unavailable for {year}")
    index = {(d, a, h): (ascore, hscore) for d, a, h, ascore, hscore in official}
    kept = []
    rejected = 0
    for r in games.itertuples(index=False):
        d = pd.Timestamp(r.datetime).strftime("%Y-%m-%d")
        key = (d, str(r.away), str(r.home))
        got = index.get(key)
        if got is None:
            # The official schedule may use the opposite presentation only for
            # unusual venue/record corrections. Never silently swap teams.
            rejected += 1
            continue
        if int(round(float(r.away_score))) != got[0] or int(round(float(r.home_score))) != got[1]:
            raise RuntimeError(
                f"official NPB result mismatch: {r.game_id} {d} {r.away}-{r.home} "
                f"SPAIA={r.away_score}-{r.home_score} NPB={got[0]}-{got[1]}"
            )
        kept.append(r._asdict())
    if rejected:
        print(f"[OFFICIAL SCHEDULE] year={year} rejected_unverified_rows={rejected}")
    return pd.DataFrame(kept, columns=games.columns).sort_values(["datetime", "game_id"]).reset_index(drop=True) if kept else games.iloc[0:0].copy()


'''
s = s.replace(anchor, insert + anchor, 1)

# Rename the original function and wrap it so all callers keep the same API.
s = s.replace("def fetch_games(year):\n", "def _fetch_games_spaia(year):\n", 1)
wrapper = r'''

def fetch_games(year):
    games = _fetch_games_spaia(year)
    return _official_validate_games(year, games)
'''
# Place wrapper immediately before checkpoint helpers.
marker2 = "\ndef load_checkpoint(year):\n"
if marker2 not in s:
    raise RuntimeError("load_checkpoint anchor not found")
s = s.replace(marker2, wrapper + marker2, 1)
P.write_text(s, encoding="utf-8")
print("[OFFICIAL PATCH] V1 applied: NPB.jp canonical schedule/result validation enabled")
