#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Add an authoritative NPB.jp schedule/result validation layer.

The granular SPAIA feed remains the efficient source for per-game enrichment,
but a completed NPB row is admitted only when its date, home team, away team,
and published score agree with the official NPB.jp monthly detail page.
For games that are not yet completed, the official page is used for schedule
identity only and no score is inferred. Missing/unparseable official data is
treated as missing data; the collector never guesses or swaps teams silently.
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


def _official_schedule_rows(year):
    """Return canonical (date, home, away, home_score, away_score, score_known)."""
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
                # NPB.jp displays the home team on the left and visitor on the
                # right: "巨人 4 - 3 DeNA". Future games have no score.
                item = re.sub(r"\s+", " ", card).strip()
                item = re.sub(r"\([^)]*\)", "", item).strip()
                if not item or item in ("-", "nan"):
                    continue
                score = re.match(r"^(.+?)\s+(\d+)\s*-\s*(\d+)\s+(.+?)$", item)
                if score:
                    home_raw, home_score, away_score, away_raw = score.groups()
                    score_known = True
                else:
                    matchup = re.match(r"^(.+?)\s*-\s*(.+?)$", item)
                    if not matchup:
                        continue
                    home_raw, away_raw = matchup.groups()
                    home_score = away_score = None
                    score_known = False
                home = official_name(home_raw.strip())
                away = official_name(away_raw.strip())
                if home not in ALIASES.values() or away not in ALIASES.values() or home == away:
                    continue
                rows.append((
                    f"{year:04d}-{mm:02d}-{dd:02d}",
                    home,
                    away,
                    int(home_score) if score_known else None,
                    int(away_score) if score_known else None,
                    score_known,
                ))
    _OFFICIAL_SCHEDULE_CACHE[year] = rows
    return rows


def _official_validate_games(year, games):
    """Validate SPAIA identity against NPB.jp without reversing home/away."""
    if games.empty:
        return games
    official = _official_schedule_rows(year)
    if not official:
        raise RuntimeError(f"official NPB schedule validation unavailable for {year}")
    index = {(d, home, away): (hs, aas, known) for d, home, away, hs, aas, known in official}
    kept = []
    rejected = 0
    score_checked = 0
    for r in games.itertuples(index=False):
        d = pd.Timestamp(r.datetime).strftime("%Y-%m-%d")
        key = (d, str(r.home), str(r.away))
        got = index.get(key)
        if got is None:
            rejected += 1
            continue
        official_home_score, official_away_score, score_known = got
        if score_known:
            score_checked += 1
            if int(round(float(r.home_score))) != official_home_score or int(round(float(r.away_score))) != official_away_score:
                raise RuntimeError(
                    f"official NPB result mismatch: {r.game_id} {d} {r.home}-{r.away} "
                    f"SPAIA={r.home_score}-{r.away_score} NPB={official_home_score}-{official_away_score}"
                )
        kept.append(r._asdict())
    if rejected:
        print(f"[OFFICIAL SCHEDULE] year={year} rejected_unverified_rows={rejected}")
    print(f"[OFFICIAL SCHEDULE] year={year} admitted={len(kept)} score_checked={score_checked}")
    if not kept:
        raise RuntimeError(f"official NPB schedule validation admitted zero games for {year}")
    return pd.DataFrame(kept, columns=games.columns).sort_values(["datetime", "game_id"]).reset_index(drop=True)


'''
s = s.replace(anchor, insert + anchor, 1)
s = s.replace("def fetch_games(year):\n", "def _fetch_games_spaia(year):\n", 1)
wrapper = r'''

def fetch_games(year):
    games = _fetch_games_spaia(year)
    return _official_validate_games(year, games)
'''
marker2 = "\ndef load_checkpoint(year):\n"
if marker2 not in s:
    raise RuntimeError("load_checkpoint anchor not found")
s = s.replace(marker2, wrapper + marker2, 1)
P.write_text(s, encoding="utf-8")
print("[OFFICIAL PATCH] V1 applied: NPB.jp schedule identity + published result validation enabled")
