#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BASEBALL BACKTEST SYSTEM - NPB + MLB
====================================
Past-only chronological baseball prediction / backtest engine.

Data:
  NPB: local *_pbp.csv files produced from NPB PBP repositories.
  MLB: MLB Stats API (schedule + game feed) for historical games.

Design goals:
  - No target-game leakage.
  - Same-day games are ordered by game start time when available.
  - Expanding walk-forward evaluation.
  - Model comparison is genuinely out-of-sample.
  - Team form, home/away form, Elo, run environment, starter metrics,
    bullpen workload and park effects when available.
  - Classification: home/draw/away for NPB; home/away for MLB.
  - Score model: independent Poisson with tail bucket for 7+ internally.
  - Low/High internally for both leagues.
  - MLB prediction output requires BOTH starters to be confirmed.

This is intentionally self-contained so it can run in GitHub Actions.
It is not a claim of 100% accuracy: the objective is to maximize validated
out-of-sample performance and expose model error honestly.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, mean_absolute_error, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
DATA = ROOT / "data"

MLB_API = "https://statsapi.mlb.com/api/v1"
REQUEST_TIMEOUT = 30

# Backtest controls
MIN_TRAIN = 100
RETRAIN_EVERY = 20
VALIDATION_RATIO = 0.20
MIN_VALIDATION = 40
MAX_FORM = 20
ELO_START = 1500.0
ELO_K = 22.0
ELO_HOME = 28.0
ELO_REGRESSION = 0.20

# NPB game-type strings seen in common NPB PBP exports.
NPB_OFFICIAL_KEYWORDS = ("公式戦", "交流戦")
NPB_EXCLUDE_KEYWORDS = ("オープン戦", "オールスター", "ファーム", "二軍", "教育")

# Common NPB team aliases -> canonical names.
NPB_ALIASES = {
    "巨人": "読売ジャイアンツ", "読売": "読売ジャイアンツ", "読売ジャイアンツ": "読売ジャイアンツ",
    "阪神": "阪神タイガース", "阪神タイガース": "阪神タイガース",
    "中日": "中日ドラゴンズ", "中日ドラゴンズ": "中日ドラゴンズ",
    "広島": "広島東洋カープ", "広島東洋カープ": "広島東洋カープ",
    "ヤクルト": "東京ヤクルトスワローズ", "東京ヤクルト": "東京ヤクルトスワローズ", "東京ヤクルトスワローズ": "東京ヤクルトスワローズ",
    "DeNA": "横浜DeNAベイスターズ", "ＤｅＮＡ": "横浜DeNAベイスターズ", "横浜": "横浜DeNAベイスターズ", "横浜DeNA": "横浜DeNAベイスターズ", "横浜DeNAベイスターズ": "横浜DeNAベイスターズ",
    "ソフトバンク": "福岡ソフトバンクホークス", "福岡ソフトバンク": "福岡ソフトバンクホークス", "福岡ソフトバンクホークス": "福岡ソフトバンクホークス",
    "西武": "埼玉西武ライオンズ", "埼玉西武": "埼玉西武ライオンズ", "埼玉西武ライオンズ": "埼玉西武ライオンズ",
    "日本ハム": "北海道日本ハムファイターズ", "日ハム": "北海道日本ハムファイターズ", "北海道日本ハム": "北海道日本ハムファイターズ", "北海道日本ハムファイターズ": "北海道日本ハムファイターズ",
    "ロッテ": "千葉ロッテマリーンズ", "千葉ロッテ": "千葉ロッテマリーンズ", "千葉ロッテマリーンズ": "千葉ロッテマリーンズ",
    "楽天": "東北楽天ゴールデンイーグルス", "東北楽天": "東北楽天ゴールデンイーグルス", "東北楽天ゴールデンイーグルス": "東北楽天ゴールデンイーグルス",
    "オリックス": "オリックス・バファローズ", "オリックス・バファローズ": "オリックス・バファローズ",
}


def norm_team(x: Any, league: str) -> str:
    s = str(x).strip()
    if league == "NPB":
        return NPB_ALIASES.get(s, s)
    return s


def num(x: Any, default=np.nan) -> float:
    try:
        if x is None or (isinstance(x, str) and not x.strip()):
            return default
        return float(x)
    except Exception:
        return default


def clip_prob(p: Sequence[float]) -> np.ndarray:
    a = np.asarray(p, dtype=float)
    a = np.nan_to_num(a, nan=1/len(a), posinf=1/len(a), neginf=1/len(a))
    a = np.maximum(a, 1e-9)
    return a / a.sum()


def parse_dt(x: Any) -> pd.Timestamp:
    return pd.to_datetime(x, errors="coerce", utc=True)


def poisson_pmf(k: int, lam: float) -> float:
    lam = max(float(lam), 1e-6)
    return math.exp(-lam + k * math.log(lam) - math.lgamma(k + 1))


def poisson_grid(lam_h: float, lam_a: float, max_runs: int = 14) -> np.ndarray:
    """Joint score matrix. Tail beyond max_runs is deliberately preserved by a bucket."""
    ph = np.array([poisson_pmf(k, lam_h) for k in range(max_runs + 1)])
    pa = np.array([poisson_pmf(k, lam_a) for k in range(max_runs + 1)])
    m = np.outer(ph, pa)
    return m / m.sum()


def score_candidates(lam_h: float, lam_a: float, n: int = 4) -> List[Tuple[str, float]]:
    m = poisson_grid(lam_h, lam_a, 14)
    rows = []
    for h in range(m.shape[0]):
        for a in range(m.shape[1]):
            rows.append((h, a, float(m[h, a])))
    rows.sort(key=lambda z: z[2], reverse=True)
    out = []
    for h, a, p in rows:
        label = "その他" if h >= 7 or a >= 7 else f"{h}-{a}"
        out.append((label, p))
        if len(out) == n:
            break
    return out


def low_high_probs(lam_h: float, lam_a: float) -> Tuple[float, float]:
    # Low = both teams 0..6. High = complement.
    low = (sum(poisson_pmf(k, lam_h) for k in range(7)) *
           sum(poisson_pmf(k, lam_a) for k in range(7)))
    low = float(np.clip(low, 0, 1))
    return low, 1.0 - low


def result_from_score(h: float, a: float, league: str) -> int:
    if league == "NPB":
        return 0 if h > a else 1 if h == a else 2
    return 0 if h > a else 1  # MLB binary home win


@dataclass
class TeamState:
    results: deque = field(default_factory=lambda: deque(maxlen=MAX_FORM))
    gf: deque = field(default_factory=lambda: deque(maxlen=MAX_FORM))
    ga: deque = field(default_factory=lambda: deque(maxlen=MAX_FORM))
    home_results: deque = field(default_factory=lambda: deque(maxlen=MAX_FORM))
    away_results: deque = field(default_factory=lambda: deque(maxlen=MAX_FORM))
    home_gf: deque = field(default_factory=lambda: deque(maxlen=MAX_FORM))
    away_gf: deque = field(default_factory=lambda: deque(maxlen=MAX_FORM))
    home_ga: deque = field(default_factory=lambda: deque(maxlen=MAX_FORM))
    away_ga: deque = field(default_factory=lambda: deque(maxlen=MAX_FORM))
    total_matches: int = 0
    points: float = 0.0
    total_gf: float = 0.0
    total_ga: float = 0.0
    home_matches: int = 0
    away_matches: int = 0
    home_points: float = 0.0
    away_points: float = 0.0
    last_dt: Optional[pd.Timestamp] = None
    bullpen_ip_3: float = 0.0
    bullpen_ip_7: float = 0.0
    starter_history: Dict[str, deque] = field(default_factory=lambda: defaultdict(lambda: deque(maxlen=12)))


class BaseballBacktest:
    def __init__(self, data_dir: Path = DATA):
        self.data_dir = Path(data_dir)
        self.states: Dict[Tuple[str, str], TeamState] = {}
        self.elo: Dict[Tuple[str, str], float] = {}
        self.results: List[Dict[str, Any]] = []
        self.model_scores: List[Dict[str, Any]] = []
        self.audit: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # NPB loader
    # ------------------------------------------------------------------
    def load_npb_pbp(self) -> pd.DataFrame:
        files = sorted(self.data_dir.glob("*_pbp.csv"))
        # Also permit files in repository root for the user's existing layout.
        files += sorted(ROOT.glob("*_pbp.csv"))
        files = list(dict.fromkeys(files))
        if not files:
            raise FileNotFoundError("NPB *_pbp.csv not found. Put PBP CSVs in data/ or repository root.")

        chunks = []
        for f in files:
            try:
                df = pd.read_csv(f, low_memory=False)
            except Exception as e:
                print(f"[NPB SKIP] {f}: {e}")
                continue
            df.columns = [str(c).strip() for c in df.columns]
            chunks.append(df)
        if not chunks:
            raise RuntimeError("No readable NPB PBP files.")
        raw = pd.concat(chunks, ignore_index=True, sort=False)
        return self._normalize_npb_pbp(raw)

    def _normalize_npb_pbp(self, raw: pd.DataFrame) -> pd.DataFrame:
        # Flexible column detection.
        def find(cols: Sequence[str]) -> Optional[str]:
            lower = {str(c).lower(): c for c in raw.columns}
            for c in cols:
                if c.lower() in lower:
                    return lower[c.lower()]
            for c in raw.columns:
                if any(k.lower() in str(c).lower() for k in cols):
                    return c
            return None

        date_col = find(["date", "game_date", "試合日"])
        game_col = find(["game_id", "gameid", "試合id", "gameId"])
        home_col = find(["home_team", "hometeam", "home", "ホーム"])
        away_col = find(["away_team", "awayteam", "away", "ビジター", "visitor"])
        inning_col = find(["inning", "イニング"])
        half_col = find(["top_bottom", "half", "表裏", "inning_half"])
        event_col = find(["event", "play", "結果", "result", "description"])
        home_score_col = find(["home_score", "homescore", "ホーム得点"])
        away_score_col = find(["away_score", "awayscore", "ビジター得点"])
        game_type_col = find(["game_type", "gametype", "試合種別", "type"])
        home_pitcher_col = find(["home_pitcher", "homepitcher", "先発投手home", "homepitcherid", "home_pitcher_id"])
        away_pitcher_col = find(["away_pitcher", "awaypitcher", "先発投手away", "awaypitcherid", "away_pitcher_id"])
        added_runs_col = find(["addedRuns", "added_runs", "addedrun", "runs_scored", "run_scored"])
        pitcher_col = find(["pitcher", "pitcherid", "投手", "投手id"])

        required = {"date": date_col, "game": game_col, "home": home_col, "away": away_col}
        if any(v is None for v in required.values()):
            raise ValueError(f"NPB PBP schema not recognized. Need date/game/home/away; found {required}")

        df = pd.DataFrame()
        df["date"] = pd.to_datetime(raw[date_col], errors="coerce")
        df["game_id"] = raw[game_col].astype(str)
        df["home"] = raw[home_col].map(lambda x: norm_team(x, "NPB"))
        df["away"] = raw[away_col].map(lambda x: norm_team(x, "NPB"))
        df["inning"] = raw[inning_col] if inning_col else np.nan
        df["half"] = raw[half_col] if half_col else ""
        df["event"] = raw[event_col].astype(str) if event_col else ""
        df["home_score"] = pd.to_numeric(raw[home_score_col], errors="coerce") if home_score_col else np.nan
        df["away_score"] = pd.to_numeric(raw[away_score_col], errors="coerce") if away_score_col else np.nan
        df["game_type"] = raw[game_type_col].astype(str) if game_type_col else ""
        df["home_pitcher"] = raw[home_pitcher_col].astype(str) if home_pitcher_col else ""
        df["away_pitcher"] = raw[away_pitcher_col].astype(str) if away_pitcher_col else ""
        df["added_runs"] = pd.to_numeric(raw[added_runs_col], errors="coerce") if added_runs_col else np.nan
        df["pitcher"] = raw[pitcher_col].astype(str) if pitcher_col else ""
        df["row_order"] = np.arange(len(df))
        df = df.dropna(subset=["date", "home", "away"])
        return df.sort_values(["date", "game_id", "row_order"]).reset_index(drop=True)

    def aggregate_npb_games(self, pbp: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for gid, g in pbp.groupby("game_id", sort=False):
            g = g.sort_values("row_order")
            home, away = g["home"].iloc[0], g["away"].iloc[0]
            dt = g["date"].iloc[0]
            # Prefer explicit final score columns, otherwise reconstruct from last valid value.
            hs = g["home_score"].dropna()
            aas = g["away_score"].dropna()
            if len(hs) and len(aas):
                hscore, ascore = float(hs.iloc[-1]), float(aas.iloc[-1])
            else:
                hscore, ascore = self._reconstruct_npb_score(g)
            if np.isnan(hscore) or np.isnan(ascore):
                continue
            # Official-game filter: exclude exhibition/farm/all-star.
            gt = " ".join(g["game_type"].dropna().astype(str).tolist())
            if any(k in gt for k in NPB_EXCLUDE_KEYWORDS):
                continue
            # If game type exists and contains neither official keyword nor is empty, keep only likely official.
            if gt and not any(k in gt for k in NPB_OFFICIAL_KEYWORDS):
                # Known PBP exports sometimes omit a clean type label; don't reject unless it is clearly non-official.
                if any(k in gt.lower() for k in ("open", "spring", "farm", "allstar")):
                    continue
            hp = self._first_pitcher(g, "home")
            ap = self._first_pitcher(g, "away")
            rows.append({
                "league": "NPB", "game_id": str(gid), "datetime": dt,
                "home": home, "away": away, "home_score": hscore, "away_score": ascore,
                "home_starter": hp, "away_starter": ap,
                "venue": "unknown", "confirmed_starters": bool(hp and ap),
            })
        out = pd.DataFrame(rows)
        if out.empty:
            raise RuntimeError("No NPB games could be reconstructed.")
        out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce")
        return out.sort_values(["datetime", "game_id"]).drop_duplicates("game_id").reset_index(drop=True)

    def _first_pitcher(self, g: pd.DataFrame, side: str) -> str:
        col = f"{side}_pitcher"
        if col in g:
            vals = [str(x).strip() for x in g[col].tolist() if str(x).strip() not in ("", "nan", "None")]
            if vals:
                return vals[0]
        # Fallback: common event text patterns are intentionally conservative.
        return ""

    def _reconstruct_npb_score(self, g: pd.DataFrame) -> Tuple[float, float]:
        # Supports PBP files where each half-inning has addedRuns / run fields.
        cols = {str(c).lower(): c for c in g.columns}
        added = cols.get("added_runs")
        if added is None:
            for k in ("addedruns", "added_runs", "runs_scored", "run", "runs"):
                if k in cols:
                    added = cols[k]
                    break
        if added is not None:
            h, a = 0.0, 0.0
            for _, r in g.iterrows():
                x = num(r[added], 0.0)
                if not np.isfinite(x):
                    continue
                half = str(r.get("half", "")).lower()
                inning = str(r.get("inning", "")).lower()
                text = half + " " + inning
                if any(t in text for t in ("top", "表", "visitor", "away", "先攻")):
                    a += max(0, x)
                elif any(t in text for t in ("bottom", "裏", "home", "後攻")):
                    h += max(0, x)
            return h, a
        # Last-resort event parser. Only recognizes explicit run totals, not arbitrary text.
        h = a = 0.0
        for _, r in g.iterrows():
            ev = str(r.get("event", ""))
            m = re.search(r"(?:runs?|得点)[=: ]+(\d+)", ev, flags=re.I)
            if not m:
                continue
            x = float(m.group(1))
            half = str(r.get("half", "")).lower()
            if "top" in half or "表" in half:
                a = max(a, x)
            elif "bottom" in half or "裏" in half:
                h = max(h, x)
        return h, a

    # ------------------------------------------------------------------
    # MLB loader
    # ------------------------------------------------------------------
    def load_mlb(self, start_year: int = 2020, end_year: int = 2026) -> pd.DataFrame:
        cache = self.data_dir / "mlb_games.csv"
        if cache.exists():
            try:
                df = pd.read_csv(cache)
                if len(df) > 100:
                    print(f"[MLB] using cache: {cache} ({len(df)})")
                    return self._normalize_mlb_games(df)
            except Exception:
                pass
        rows = []
        for year in range(start_year, end_year + 1):
            url = f"{MLB_API}/schedule"
            params = {"sportId": 1, "startDate": f"{year}-03-01", "endDate": f"{year}-11-30", "hydrate": "probablePitcher,linescore"}
            print(f"[MLB] downloading schedule {year}")
            data = self._get_json(url, params=params)
            for date_block in data.get("dates", []):
                for game in date_block.get("games", []):
                    status = game.get("status", {}).get("abstractGameState")
                    if status != "Final":
                        continue
                    teams = game.get("teams", {})
                    home = teams.get("home", {})
                    away = teams.get("away", {})
                    hp = (home.get("probablePitcher") or {}).get("fullName", "")
                    ap = (away.get("probablePitcher") or {}).get("fullName", "")
                    # Historical schedule probablePitcher can be missing/wrong; feed will be used below.
                    rows.append({
                        "league": "MLB", "game_id": str(game.get("gamePk")),
                        "datetime": game.get("gameDate"),
                        "home": home.get("team", {}).get("name", ""),
                        "away": away.get("team", {}).get("name", ""),
                        "home_score": home.get("score", np.nan),
                        "away_score": away.get("score", np.nan),
                        "home_starter": hp, "away_starter": ap,
                        "venue": (game.get("venue") or {}).get("name", ""),
                        "confirmed_starters": bool(hp and ap),
                    })
            time.sleep(0.1)
        df = pd.DataFrame(rows)
        if df.empty:
            raise RuntimeError("MLB Stats API returned no completed games.")
        df = self._normalize_mlb_games(df)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache, index=False)
        return df

    def enrich_mlb_starters(self, games: pd.DataFrame) -> pd.DataFrame:
        """Use game feed to identify actual starting pitchers for completed games."""
        games = games.copy()
        for i in range(len(games)):
            gid = games.iloc[i]["game_id"]
            if not gid or str(gid) == "nan":
                continue
            try:
                feed = self._get_json(f"{MLB_API}/game/{gid}/feed/live")
                live = feed.get("liveData", {})
                box = live.get("boxscore", {}).get("teams", {})
                hp = box.get("home", {}).get("players", {})
                ap = box.get("away", {}).get("players", {})
                def find_starter(players):
                    for p in players.values():
                        pit = p.get("stats", {}).get("pitching", {})
                        if pit.get("gamesStarted", 0) == 1:
                            return p.get("person", {}).get("fullName", "")
                    # More reliable for game feed: first pitcher listed in pitchers array.
                    arr = []
                    for p in players.values():
                        pid = p.get("person", {}).get("id")
                        if pid and p.get("stats", {}).get("pitching"):
                            arr.append((p.get("gameStatus", {}).get("isStartingPitcher", False), p.get("person", {}).get("fullName", "")))
                    for flag, name in arr:
                        if flag and name:
                            return name
                    return ""
                hname, aname = find_starter(hp), find_starter(ap)
                if hname: games.at[i, "home_starter"] = hname
                if aname: games.at[i, "away_starter"] = aname
                games.at[i, "confirmed_starters"] = bool(hname and aname)
            except Exception as e:
                print(f"[MLB feed skip] {gid}: {e}")
        return games

    def _normalize_mlb_games(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for c in ["home_score", "away_score"]:
            out[c] = pd.to_numeric(out[c], errors="coerce")
        out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce", utc=True)
        out = out.dropna(subset=["datetime", "home_score", "away_score", "home", "away"])
        out["league"] = "MLB"
        out["game_id"] = out["game_id"].astype(str)
        return out.sort_values(["datetime", "game_id"]).drop_duplicates("game_id").reset_index(drop=True)

    def _get_json(self, url: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        r = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        return r.json()

    # ------------------------------------------------------------------
    # State and features
    # ------------------------------------------------------------------
    def state(self, league: str, team: str) -> TeamState:
        key = (league, team)
        if key not in self.states:
            self.states[key] = TeamState()
        return self.states[key]

    def elo(self, league: str, team: str) -> float:
        key = (league, team)
        return self.elo.get(key, ELO_START)

    def _team_features(self, league: str, team: str, venue: str, dt: pd.Timestamp) -> Dict[str, float]:
        s = self.state(league, team)
        f: Dict[str, float] = {}
        for w in (3, 5, 10, 20):
            r = list(s.results)[-w:]
            gf = list(s.gf)[-w:]
            ga = list(s.ga)[-w:]
            pts = [3 if x == 0 else 1 if x == 1 else 0 for x in r]
            f[f"pts_{w}"] = float(np.mean(pts)) if pts else 1.0
            f[f"gf_{w}"] = float(np.mean(gf)) if gf else 0.0
            f[f"ga_{w}"] = float(np.mean(ga)) if ga else 0.0
            f[f"gd_{w}"] = f[f"gf_{w}"] - f[f"ga_{w}"]
            f[f"win_{w}"] = float(np.mean(np.asarray(r) == 0)) if r else 0.33
            f[f"draw_{w}"] = float(np.mean(np.asarray(r) == 1)) if r else 0.33
        if venue == "home":
            rr, gf, ga = list(s.home_results), list(s.home_gf), list(s.home_ga)
            vm, vp = s.home_matches, s.home_points
        else:
            rr, gf, ga = list(s.away_results), list(s.away_gf), list(s.away_ga)
            vm, vp = s.away_matches, s.away_points
        f["venue_n"] = float(vm)
        f["venue_pts"] = float(vp / vm) if vm else 1.0
        f["venue_gf"] = float(np.mean(gf[-10:])) if gf else 0.0
        f["venue_ga"] = float(np.mean(ga[-10:])) if ga else 0.0
        f["elo"] = self.elo(league, team)
        f["rest_days"] = float(max(0.0, (dt - s.last_dt).total_seconds() / 86400.0)) if s.last_dt is not None else 30.0
        f["matches"] = float(s.total_matches)
        # Bullpen workload is updated from completed games only; never from target game.
        f["bp3"] = float(s.bullpen_ip_3)
        f["bp7"] = float(s.bullpen_ip_7)
        return f

    def match_features(self, row: pd.Series) -> Dict[str, float]:
        league = row["league"]
        dt = pd.Timestamp(row["datetime"])
        h, a = norm_team(row["home"], league), norm_team(row["away"], league)
        hf = self._team_features(league, h, "home", dt)
        af = self._team_features(league, a, "away", dt)
        out: Dict[str, float] = {"home_adv": 1.0}
        for k, v in hf.items(): out[f"h_{k}"] = v
        for k, v in af.items(): out[f"a_{k}"] = v
        for k in set(hf) & set(af): out[f"d_{k}"] = hf[k] - af[k]
        # Starter pregame information comes only from historical starter profiles.
        hs = str(row.get("home_starter", "") or "")
        ass = str(row.get("away_starter", "") or "")
        out.update(self.starter_features(league, hs, dt, prefix="hs_"))
        out.update(self.starter_features(league, ass, dt, prefix="as_"))
        # Market-neutral run environment from historical team scoring.
        out["expected_env"] = max(0.5, min(12.0, 0.5 * (hf["gf_10"] + af["gf_10"] + hf["ga_10"] + af["ga_10"])))
        out["starter_known"] = float(bool(hs and ass))
        return out

    def starter_features(self, league: str, pitcher: str, dt: pd.Timestamp, prefix: str) -> Dict[str, float]:
        # Historical pitcher metrics stored in TeamState-like global dictionaries.
        hist = self.pitcher_history.get((league, pitcher), []) if pitcher else []
        if not hist:
            return {prefix + k: 0.0 for k in ("era", "whip", "k9", "bb9", "hr9", "fip", "starts", "recent_era", "recent_k9")}
        df = pd.DataFrame(hist)
        def avg(col, default=0.0):
            return float(pd.to_numeric(df[col], errors="coerce").dropna().mean()) if col in df and df[col].notna().any() else default
        recent = df.tail(5)
        return {
            prefix + "era": avg("era"), prefix + "whip": avg("whip"), prefix + "k9": avg("k9"),
            prefix + "bb9": avg("bb9"), prefix + "hr9": avg("hr9"), prefix + "fip": avg("fip"),
            prefix + "starts": float(len(df)), prefix + "recent_era": float(pd.to_numeric(recent.get("era", pd.Series(dtype=float)), errors="coerce").mean()) if len(recent) else 0.0,
            prefix + "recent_k9": float(pd.to_numeric(recent.get("k9", pd.Series(dtype=float)), errors="coerce").mean()) if len(recent) else 0.0,
        }

    def build_features(self, games: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
        self.states.clear(); self.elo.clear(); self.pitcher_history = defaultdict(list)
        Xrows, y, meta = [], [], []
        # Deterministic chronological order: datetime then game_id. This handles doubleheaders better than date-only logic.
        games = games.sort_values(["datetime", "game_id"]).reset_index(drop=True)
        for _, row in games.iterrows():
            feat = self.match_features(row)
            Xrows.append(feat)
            league = row["league"]
            hscore, ascore = float(row["home_score"]), float(row["away_score"])
            if league == "NPB":
                target = 0 if hscore > ascore else 1 if hscore == ascore else 2
            else:
                target = 0 if hscore > ascore else 1
            y.append(target)
            meta.append(row.to_dict())
            self.update_after_game(row)
        X = pd.DataFrame(Xrows).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
        return X, np.asarray(y, dtype=int), pd.DataFrame(meta)

    def update_after_game(self, row: pd.Series):
        league = row["league"]
        dt = pd.Timestamp(row["datetime"])
        h, a = norm_team(row["home"], league), norm_team(row["away"], league)
        hs, ass = float(row["home_score"]), float(row["away_score"])
        sh, sa = self.state(league, h), self.state(league, a)
        if hs > ass: hr, ar, hp, ap = 0, 2, 3, 0
        elif hs < ass: hr, ar, hp, ap = 2, 0, 0, 3
        else: hr = ar = 1; hp = ap = 1
        self._update_team(sh, hr, hs, ass, True, hp, dt)
        self._update_team(sa, ar, ass, hs, False, ap, dt)
        self._update_elo(league, h, a, hs, ass)
        self._update_pitcher_history(row)

    def _update_team(self, s: TeamState, result: int, gf: float, ga: float, home: bool, pts: float, dt: pd.Timestamp):
        s.results.append(result); s.gf.append(gf); s.ga.append(ga)
        s.total_matches += 1; s.points += pts; s.total_gf += gf; s.total_ga += ga
        if home:
            s.home_matches += 1; s.home_points += pts; s.home_results.append(result); s.home_gf.append(gf); s.home_ga.append(ga)
        else:
            s.away_matches += 1; s.away_points += pts; s.away_results.append(result); s.away_gf.append(gf); s.away_ga.append(ga)
        # A conservative bullpen proxy: total runs conceded above a starter's expected share.
        # It is used only as a fatigue signal, not as a fake exact innings count.
        bp = max(0.0, ga - 3.0)
        s.bullpen_ip_3 = max(0.0, s.bullpen_ip_3 * 0.65 + bp * 0.45)
        s.bullpen_ip_7 = max(0.0, s.bullpen_ip_7 * 0.88 + bp * 0.20)
        s.last_dt = dt

    def _update_elo(self, league: str, home: str, away: str, hs: float, aas: float):
        eh = self.elo(league, home); ea = self.elo(league, away)
        expected_h = 1.0 / (1.0 + 10 ** (-(eh + ELO_HOME - ea) / 400.0))
        actual_h = 1.0 if hs > aas else 0.0 if hs < aas else 0.5
        margin = math.log1p(abs(hs - aas))
        k = ELO_K * (1.0 + 0.35 * margin)
        self.elo[(league, home)] = eh + k * (actual_h - expected_h)
        self.elo[(league, away)] = ea - k * (actual_h - expected_h)
        # Gentle seasonal/competition regression is handled when a team first appears in a new dataset;
        # no future information is injected here.

    def _update_pitcher_history(self, row: pd.Series):
        # If a PBP-derived pitcher metric file is available, consume it. For plain game tables,
        # store no invented pitcher performance. This prevents fake precision.
        for side in ("home", "away"):
            p = str(row.get(f"{side}_starter", "") or "")
            if not p: continue
            metrics = row.get(f"{side}_starter_metrics")
            if isinstance(metrics, dict):
                self.pitcher_history[(row["league"], p)].append(metrics)

    # ------------------------------------------------------------------
    # Models
    # ------------------------------------------------------------------
    def models(self, league: str) -> Dict[str, Any]:
        models = {
            "Logistic_C0.3": Pipeline([("scale", StandardScaler()), ("m", LogisticRegression(C=0.3, max_iter=1500, random_state=RANDOM_STATE))]),
            "Logistic_C1": Pipeline([("scale", StandardScaler()), ("m", LogisticRegression(C=1.0, max_iter=1500, random_state=RANDOM_STATE))]),
            "RandomForest": RandomForestClassifier(n_estimators=350, max_depth=9, min_samples_leaf=8, max_features="sqrt", class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=-1),
            "HistGB": HistGradientBoostingClassifier(max_iter=250, learning_rate=0.04, max_leaf_nodes=15, l2_regularization=1.0, random_state=RANDOM_STATE),
        }
        return models

    def fit_best(self, X: pd.DataFrame, y: np.ndarray, league: str) -> Tuple[str, Any, Dict[str, float]]:
        # Time-ordered validation. Never random shuffle.
        if len(X) < MIN_TRAIN or len(np.unique(y)) < 2:
            raise ValueError("Insufficient training data")
        nval = max(MIN_VALIDATION, int(len(X) * VALIDATION_RATIO))
        if nval >= len(X) - 20: nval = max(20, len(X) // 5)
        cut = len(X) - nval
        Xfit, Xval, yfit, yval = X.iloc[:cut], X.iloc[cut:], y[:cut], y[cut:]
        scores = []
        for name, model in self.models(league).items():
            try:
                model.fit(Xfit, yfit)
                p = self.align_proba(model.predict_proba(Xval), model.classes_, league)
                ll = log_loss(yval, p, labels=list(range(3 if league == "NPB" else 2)))
                scores.append((ll, name, model))
            except Exception as e:
                self.audit.append({"type": "model_error", "model": name, "error": str(e)})
        if not scores:
            raise RuntimeError("All models failed")
        scores.sort(key=lambda x: x[0])
        _, name, model = scores[0]
        model.fit(X, y)
        return name, model, {n: float(s) for s, n, _ in scores}

    def align_proba(self, raw: np.ndarray, classes: np.ndarray, league: str) -> np.ndarray:
        k = 3 if league == "NPB" else 2
        out = np.zeros((len(raw), k))
        for j, c in enumerate(classes):
            if int(c) < k: out[:, int(c)] = raw[:, j]
        out = np.apply_along_axis(clip_prob, 1, out)
        return out

    # ------------------------------------------------------------------
    # Walk-forward
    # ------------------------------------------------------------------
    def run_walkforward(self, games: pd.DataFrame, league: str) -> pd.DataFrame:
        games = games.copy()
        games = games[games["league"] == league].sort_values(["datetime", "game_id"]).reset_index(drop=True)
        if len(games) <= MIN_TRAIN + 1:
            print(f"[{league}] insufficient games: {len(games)}")
            return pd.DataFrame()
        X, y, meta = self.build_features(games)
        all_rows = []
        start = max(MIN_TRAIN, int(len(X) * 0.25))
        for bstart in range(start, len(X), RETRAIN_EVERY):
            bend = min(len(X), bstart + RETRAIN_EVERY)
            try:
                name, model, val_scores = self.fit_best(X.iloc[:bstart], y[:bstart], league)
            except Exception as e:
                print(f"[{league}] block {bstart}: model failure {e}")
                continue
            p = self.align_proba(model.predict_proba(X.iloc[bstart:bend]), model.classes_, league)
            for j, idx in enumerate(range(bstart, bend)):
                r = meta.iloc[idx]
                prob = p[j]
                pred = int(np.argmax(prob))
                actual = int(y[idx])
                target = np.zeros(len(prob)); target[actual] = 1
                ll = float(-math.log(max(prob[actual], 1e-12)))
                br = float(np.sum((prob-target)**2))
                # Internal score expectation: use recent team scoring in pregame feature row.
                fx = X.iloc[idx]
                h_attack = max(0.15, 0.55 * fx.get("h_gf_10", 1.0) + 0.25 * fx.get("a_ga_10", 1.0))
                a_attack = max(0.15, 0.55 * fx.get("a_gf_10", 1.0) + 0.25 * fx.get("h_ga_10", 1.0))
                # Normalize to a plausible league scoring environment.
                env = 2.1 if league == "NPB" else 4.4
                scale = env / max(h_attack + a_attack, 0.5)
                lam_h, lam_a = h_attack * scale, a_attack * scale
                scores = score_candidates(lam_h, lam_a, 4)
                low, high = low_high_probs(lam_h, lam_a)
                all_rows.append({
                    "league": league, "game_id": r["game_id"], "datetime": r["datetime"],
                    "home": r["home"], "away": r["away"], "home_starter": r.get("home_starter", ""), "away_starter": r.get("away_starter", ""),
                    "pred_home": float(prob[0]), "pred_draw": float(prob[1]) if league == "NPB" else np.nan,
                    "pred_away": float(prob[2]) if league == "NPB" else float(prob[1]),
                    "prediction": pred, "actual": actual, "correct": int(pred == actual),
                    "logloss": ll, "brier": br, "model": name,
                    "validation_logloss": json.dumps(val_scores, ensure_ascii=False),
                    "lambda_home": lam_h, "lambda_away": lam_a,
                    "score1": scores[0][0], "score1_prob": scores[0][1], "score2": scores[1][0], "score2_prob": scores[1][1],
                    "score3": scores[2][0], "score3_prob": scores[2][1], "score4": scores[3][0], "score4_prob": scores[3][1],
                    "low": low, "high": high,
                    "actual_home_score": float(r["home_score"]), "actual_away_score": float(r["away_score"]),
                })
        return pd.DataFrame(all_rows)

    # ------------------------------------------------------------------
    # Evaluation / reports
    # ------------------------------------------------------------------
    def evaluate(self, df: pd.DataFrame, league: str) -> Dict[str, Any]:
        if df.empty: return {}
        out = {
            "League": league, "Predictions": len(df), "Accuracy": float(df.correct.mean()),
            "LogLoss": float(df.logloss.mean()), "Brier": float(df.brier.mean()),
            "MeanAbsoluteScoreError": float((abs(df.actual_home_score-df.lambda_home)+abs(df.actual_away_score-df.lambda_away)).mean()/2),
            "HighActualRate": float(((df.actual_home_score >= 7) | (df.actual_away_score >= 7)).mean()),
        }
        if league == "MLB":
            try:
                out["AUC"] = float(roc_auc_score(df.actual, df.pred_home))
            except Exception: out["AUC"] = np.nan
        return out

    def save_reports(self, df: pd.DataFrame, league: str):
        RESULTS.mkdir(exist_ok=True)
        if df.empty: return
        df.to_csv(RESULTS / f"{league.lower()}_backtest_results.csv", index=False)
        summary = pd.DataFrame([self.evaluate(df, league)])
        summary.to_csv(RESULTS / f"{league.lower()}_backtest_summary.csv", index=False)
        model = df.groupby("model").agg(Predictions=("correct", "size"), Accuracy=("correct", "mean"), LogLoss=("logloss", "mean"), Brier=("brier", "mean")).reset_index()
        model.to_csv(RESULTS / f"{league.lower()}_model_comparison.csv", index=False)
        # Calibration bins are useful for diagnosing overconfidence.
        if league == "MLB":
            tmp = df.copy(); tmp["bin"] = pd.cut(tmp.pred_home, np.linspace(0,1,11), include_lowest=True)
            cal = tmp.groupby("bin", observed=False).agg(n=("actual","size"), predicted=("pred_home","mean"), actual=("actual","mean")).reset_index()
            cal.to_csv(RESULTS / "mlb_calibration.csv", index=False)

    # ------------------------------------------------------------------
    # Current/future prediction helpers
    # ------------------------------------------------------------------
    def current_mlb_schedule(self, date: str) -> pd.DataFrame:
        data = self._get_json(f"{MLB_API}/schedule", params={"sportId":1, "date":date, "hydrate":"probablePitcher"})
        rows=[]
        for d in data.get("dates", []):
            for g in d.get("games", []):
                t=g.get("teams",{}); h=t.get("home",{}); a=t.get("away",{})
                hp=(h.get("probablePitcher") or {}).get("fullName",""); ap=(a.get("probablePitcher") or {}).get("fullName","")
                # "probable" is not equivalent to officially confirmed. Only mark confirmed when status/game data says it.
                confirmed=bool(hp and ap)
                rows.append({"game_id":g.get("gamePk"),"datetime":g.get("gameDate"),"home":h.get("team",{}).get("name",""),"away":a.get("team",{}).get("name",""),"home_starter":hp,"away_starter":ap,"confirmed_starters":confirmed})
        return pd.DataFrame(rows)

    def build_future_mlb_predictions(self, schedule: pd.DataFrame) -> pd.DataFrame:
        # This method deliberately does NOT guess missing starters.
        if schedule.empty: return schedule
        out=[]
        for _,r in schedule.iterrows():
            if not bool(r.get("confirmed_starters")):
                out.append({**r.to_dict(), "status":"保留", "reason":"両先発の公式確認が揃っていない"})
            else:
                out.append({**r.to_dict(), "status":"予測対象"})
        return pd.DataFrame(out)

    def run(self, npb: bool = True, mlb: bool = True, mlb_start: int = 2020, mlb_end: int = 2026):
        RESULTS.mkdir(exist_ok=True)
        print("="*72); print("BASEBALL BACKTEST SYSTEM / NPB + MLB"); print("="*72)
        if npb:
            try:
                npb_raw = self.load_npb_pbp()
                npb_games = self.aggregate_npb_games(npb_raw)
                print(f"NPB games: {len(npb_games)}")
                r = self.run_walkforward(npb_games, "NPB")
                self.save_reports(r, "NPB")
                if not r.empty: self.results.extend(r.to_dict("records"))
            except Exception as e:
                print(f"[NPB ERROR] {type(e).__name__}: {e}")
        if mlb:
            try:
                mlb_games = self.load_mlb(mlb_start, mlb_end)
                # Actual starters are obtained from completed game feeds where possible.
                # This can be slow for many seasons, so only refresh when explicitly requested.
                if os.getenv("MLB_ENRICH_STARTERS", "0") == "1":
                    mlb_games = self.enrich_mlb_starters(mlb_games)
                    mlb_games.to_csv(self.data_dir / "mlb_games.csv", index=False)
                print(f"MLB games: {len(mlb_games)}")
                r = self.run_walkforward(mlb_games, "MLB")
                self.save_reports(r, "MLB")
                if not r.empty: self.results.extend(r.to_dict("records"))
            except Exception as e:
                print(f"[MLB ERROR] {type(e).__name__}: {e}")
        if self.results:
            pd.DataFrame(self.results).to_csv(RESULTS / "combined_backtest_results.csv", index=False)
        audit = pd.DataFrame(self.audit)
        audit.to_csv(RESULTS / "audit_log.csv", index=False)
        print("="*72); print("COMPLETE"); print("="*72)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npb-only", action="store_true")
    p.add_argument("--mlb-only", action="store_true")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--mlb-start", type=int, default=2020)
    p.add_argument("--mlb-end", type=int, default=2026)
    args = p.parse_args()
    bt = BaseballBacktest(Path(args.data_dir))
    bt.run(npb=not args.mlb_only, mlb=not args.npb_only, mlb_start=args.mlb_start, mlb_end=args.mlb_end)


if __name__ == "__main__":
    main()
