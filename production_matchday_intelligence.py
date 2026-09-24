#!/usr/bin/env python3
"""Free, fail-closed NPB Matchday Intelligence + shadow prediction runner.

Pipeline:
  official schedule -> official announced starters -> roster notices ->
  SPAIA lineup snapshot -> Open-Meteo forecast -> rest/travel state ->
  incumbent ensemble prediction -> research-only dynamic-routing candidate.

The routing candidate is NEVER promoted here. Production output remains the
incumbent model unless the repository's explicit production gates later accept
it through a separate controlled change.
"""
from __future__ import annotations

import hashlib
import html as htmlmod
import io
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
from lxml import html as lxml_html

from baseball_backtest import BaseballBacktest
from research.drift_uncertainty_routing import (
    ExpertCalibrationBank,
    RoutingConfig,
    feature_drift_score,
    mix_expert_probabilities,
    route_experts,
)
from research.conformal_uncertainty import uncertainty_summary
from research.matchday_intelligence import ContextKind, Observation
from research.matchday_reforecast import reforecast

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
DATA = ROOT / "data"
NPB = "https://npb.jp"
SPAIA = "https://spaia.jp/baseball/npb/api"
WEATHER = "https://api.open-meteo.com/v1/forecast"
JST = ZoneInfo("Asia/Tokyo")
TIMEOUT = 25

ALIASES = {
    "巨人":"読売ジャイアンツ","読売":"読売ジャイアンツ","読売ジャイアンツ":"読売ジャイアンツ",
    "阪神":"阪神タイガース","阪神タイガース":"阪神タイガース","中日":"中日ドラゴンズ","中日ドラゴンズ":"中日ドラゴンズ",
    "広島":"広島東洋カープ","広島東洋":"広島東洋カープ","広島東洋カープ":"広島東洋カープ",
    "ヤクルト":"東京ヤクルトスワローズ","東京ヤクルト":"東京ヤクルトスワローズ","東京ヤクルトスワローズ":"東京ヤクルトスワローズ",
    "DeNA":"横浜DeNAベイスターズ","ＤｅＮＡ":"横浜DeNAベイスターズ","横浜":"横浜DeNAベイスターズ","横浜DeNA":"横浜DeNAベイスターズ","横浜DeNAベイスターズ":"横浜DeNAベイスターズ",
    "ソフトバンク":"福岡ソフトバンクホークス","福岡ソフトバンク":"福岡ソフトバンクホークス","福岡ソフトバンクホークス":"福岡ソフトバンクホークス",
    "西武":"埼玉西武ライオンズ","埼玉西武":"埼玉西武ライオンズ","埼玉西武ライオンズ":"埼玉西武ライオンズ",
    "日本ハム":"北海道日本ハムファイターズ","日ハム":"北海道日本ハムファイターズ","北海道日本ハム":"北海道日本ハムファイターズ","北海道日本ハムファイターズ":"北海道日本ハムファイターズ",
    "ロッテ":"千葉ロッテマリーンズ","千葉ロッテ":"千葉ロッテマリーンズ","千葉ロッテマリーンズ":"千葉ロッテマリーンズ",
    "楽天":"東北楽天ゴールデンイーグルス","東北楽天":"東北楽天ゴールデンイーグルス","東北楽天ゴールデンイーグルス":"東北楽天ゴールデンイーグルス",
    "オリックス":"オリックス・バファローズ","オリックス・バファローズ":"オリックス・バファローズ",
}
PARKS = {
    "神　宮":(35.6827,139.6841),"神宮":(35.6827,139.6841),"東京ドーム":(35.7056,139.7519),
    "横　浜":(35.4431,139.6400),"横浜":(35.4431,139.6400),"バンテリンドーム":(35.1859,136.9470),
    "マツダスタジアム":(34.3916,132.4848),"甲子園":(34.7214,135.3616),"エスコンＦ":(43.0151,141.4094),
    "ベルーナドーム":(35.7684,139.4745),"ZOZOマリン":(35.6456,140.0307),"楽天モバイル":(38.2560,140.9014),
    "京セラD大阪":(34.6694,135.4761),"ほっと神戸":(34.6795,135.0980),"みずほPayPay":(33.5950,130.3620),
}
TEAM_NAMES = set(ALIASES.values())


def norm_team(x):
    return ALIASES.get(str(x or "").strip(), str(x or "").strip())


def get(url, params=None, attempts=4):
    last = None
    for i in range(attempts):
        try:
            r = requests.get(url, params=params, timeout=TIMEOUT, headers={"User-Agent":"Baseball-Matchday-Research/1.0"})
            r.raise_for_status()
            return r
        except Exception as exc:
            last = exc
            if i + 1 < attempts:
                import time
                time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"request failed: {url}: {last}")


def fetch_schedule(year: int, target_date: str):
    # Official monthly schedule is the identity source.
    url = f"{NPB}/games/{year}/schedule_{int(target_date[5:7]):02d}_detail.html"
    doc = lxml_html.fromstring(get(url).content)
    rows = []
    for table in pd.read_html(io.BytesIO(get(url).content)):
        if "月日" not in table.columns or "対戦カード" not in table.columns:
            continue
        venue_col = "球場・開始時間" if "球場・開始時間" in table.columns else None
        for _, rec in table.iterrows():
            date_text = str(rec.get("月日", ""))
            if not re.search(rf"{int(target_date[5:7])}/({int(target_date[8:10]):02d}|{int(target_date[8:10])})", date_text):
                continue
            card = re.sub(r"\s+", " ", str(rec.get("対戦カード", ""))).strip()
            if not card or "-" not in card or any(x in card for x in ("中止","試合前")):
                continue
            # Historical/final rows contain a numeric score around the hyphen;
            # only unsolved matchups are eligible for current prediction.
            if re.search(r"\d+\s*-\s*\d+", card):
                continue
            m = re.match(r"^(.+?)\s+(?:\d+\s+-\s+\d+|-)\s+(.+?)$", card)
            if not m:
                continue
            home, away = norm_team(m.group(1)), norm_team(m.group(2))
            if home not in TEAM_NAMES or away not in TEAM_NAMES:
                continue
            venue_text = re.sub(r"\s+", " ", str(rec.get(venue_col, ""))) if venue_col else ""
            tm = re.search(r"(\d{1,2}):(\d{2})", venue_text)
            hour, minute = (int(tm.group(1)), int(tm.group(2))) if tm else (18, 0)
            venue = venue_text.split()[0] if venue_text else "unknown"
            rows.append({
                "date": target_date, "home": home, "away": away, "venue": venue,
                "hour": hour, "minute": minute, "official_schedule": True,
            })
    # Deduplicate stable identity.
    out = {}
    for r in rows:
        out[(r["home"],r["away"],r["hour"],r["minute"],r["venue"])] = r
    return list(out.values())


def fetch_spaia_game_ids(year: int, target_date: str):
    """Resolve SPAIA game IDs, with a public-page fallback.

    The structured schedule endpoint is preferred. If it returns no usable
    pair, the public daily SPAIA page is scanned for /game/<id> links and team
    names around each link. This keeps current lineup retrieval free and avoids
    depending on undocumented API response shape.
    """
    result = {}
    try:
        raw = get(f"{SPAIA}/schedules", {"Year": year}).json()
        for g in raw if isinstance(raw, list) else []:
            d = pd.to_datetime(
                g.get("datetime") or g.get("DateJPN") or g.get("date"),
                errors="coerce",
            )
            if pd.isna(d) or d.strftime("%Y-%m-%d") != target_date:
                continue
            home = norm_team(
                g.get("HTeamNameS") or g.get("home_team_short_name")
                or g.get("homeTeam") or g.get("HomeTeamName")
            )
            away = norm_team(
                g.get("VTeamNameS") or g.get("away_team_short_name")
                or g.get("visitorTeam") or g.get("AwayTeamName")
            )
            gid = str(
                g.get("GameID") or g.get("gameId")
                or g.get("game_id") or g.get("gamePk") or ""
            ).strip()
            if home in TEAM_NAMES and away in TEAM_NAMES and gid:
                result[(home, away)] = gid
    except Exception:
        pass

    missing_pairs = [
        (h, a) for h, a in (
            ("広島東洋カープ","読売ジャイアンツ"),
            ("北海道日本ハムファイターズ","東北楽天ゴールデンイーグルス"),
            ("阪神タイガース","横浜DeNAベイスターズ"),
            ("東京ヤクルトスワローズ","中日ドラゴンズ"),
            ("千葉ロッテマリーンズ","埼玉西武ライオンズ"),
            ("福岡ソフトバンクホークス","オリックス・バファローズ"),
        ) if (h, a) not in result
    ]

    if not missing_pairs:
        return result

    try:
        url = f"https://spaia.jp/baseball/npb/?date={target_date.replace('-', '')}"
        page = get(url).content
        doc = lxml_html.fromstring(page)
        short_alias = {
            "広島東洋カープ": ("広島","広島東洋"),
            "読売ジャイアンツ": ("巨人","読売"),
            "北海道日本ハムファイターズ": ("日本ハム","日ハム"),
            "東北楽天ゴールデンイーグルス": ("楽天","東北楽天"),
            "阪神タイガース": ("阪神",),
            "横浜DeNAベイスターズ": ("DeNA","横浜"),
            "東京ヤクルトスワローズ": ("ヤクルト",),
            "中日ドラゴンズ": ("中日",),
            "千葉ロッテマリーンズ": ("ロッテ",),
            "埼玉西武ライオンズ": ("西武",),
            "福岡ソフトバンクホークス": ("ソフトバンク",),
            "オリックス・バファローズ": ("オリックス",),
        }
        for a in doc.xpath('//a[contains(@href,"/baseball/npb/game/")]'):
            href = str(a.get("href") or "")
            m = re.search(r"/baseball/npb/game/(\d+)", href)
            if not m:
                continue
            gid = m.group(1)
            parent = a.getparent()
            context = " ".join(
                str(x) for x in (
                    a.text_content(),
                    parent.text_content() if parent is not None else "",
                    parent.getparent().text_content() if parent is not None and parent.getparent() is not None else "",
                ) if x
            )
            for home, away in missing_pairs:
                if any(tok in context for tok in short_alias[home]) and any(tok in context for tok in short_alias[away]):
                    result[(home, away)] = gid
        return result
    except Exception:
        return result


def fetch_official_starters(target_date: str):
    """Parse only the NPB official starter block for the requested date.

    NPB's public starter page can roll to the next day's announced starters
    after the current slate is complete. The target-date heading is therefore
    a hard safety gate; no heading match means UNKNOWN.
    """
    r = get(f"{NPB}/announcement/starter/")
    doc = lxml_html.fromstring(r.content)
    target_label = f"{int(target_date[5:7])}月{int(target_date[8:10])}日の予告先発投手"

    heading = None
    for node in doc.xpath("//h1|//h2|//h3|//h4|//strong"):
        txt = re.sub(r"\s+", " ", "".join(node.itertext())).strip()
        if target_label in txt:
            heading = node
            break
    if heading is None:
        return {}

    starters = {}
    current_team = None
    for node in heading.xpath("following::*"):
        tag = getattr(node, "tag", None)
        if tag == "img":
            alt = str(node.get("alt") or "").strip()
            team = norm_team(alt)
            if team in TEAM_NAMES:
                current_team = team
                continue
        if tag == "a" and current_team:
            href = str(node.get("href") or "")
            txt = re.sub(r"\s+", " ", "".join(node.itertext())).strip()
            if txt and ("/bis/players/" in href or "/player/" in href):
                starters.setdefault(current_team, txt)
                current_team = None
        if tag in {"h1","h2","h3","h4"} and node is not heading:
            txt = re.sub(r"\s+", " ", "".join(node.itertext())).strip()
            if "予告先発投手" in txt:
                break
    return starters


def fetch_roster_notice(target_date: str):
    """Read official same-day NPB registration/removal notices.

    Registration/removal is treated strictly as an availability signal. The
    code does not infer injury diagnosis or severity from a roster move.
    """
    r = get(f"{NPB}/announcement/roster/")
    doc = lxml_html.fromstring(r.content)
    marker = f"{int(target_date[5:7])}月{int(target_date[8:10])}日の出場選手登録"
    root = None
    for node in doc.xpath("//h1|//h2|//h3|//h4"):
        txt = re.sub(r"\s+", " ", "".join(node.itertext())).strip()
        if marker in txt:
            root = node
            break
    text = re.sub(r"\s+", " ", doc.text_content())
    pos = text.find(marker)
    snippet = text[pos:pos+8000] if pos >= 0 else text[:8000]

    events = []
    if root is not None:
        current_status = None
        current_league = None
        for node in root.xpath("following::*"):
            tag = getattr(node, "tag", None)
            if tag in {"h3","h4","h5"}:
                title = re.sub(r"\s+", " ", "".join(node.itertext())).strip()
                if "セントラル・リーグ" in title:
                    current_league = "CENTRAL"
                elif "パシフィック・リーグ" in title:
                    current_league = "PACIFIC"
                elif title == "出場選手登録":
                    current_status = "REGISTERED"
                elif title in {"出場選手登録抹消","出場選手登録削除"}:
                    current_status = "REMOVED"
                elif "出場選手一覧" in title:
                    current_status = None
            if tag == "tr" and current_status:
                cells = [
                    re.sub(r"\s+", " ", "".join(cell.itertext())).strip()
                    for cell in node.xpath("./th|./td")
                ]
                if len(cells) >= 4 and cells[0] in TEAM_NAMES:
                    number = cells[2]
                    name = cells[3]
                    if name and name not in {"選手名", "-"}:
                        events.append({
                            "status": current_status,
                            "league": current_league,
                            "team": cells[0],
                            "position": cells[1],
                            "number": number,
                            "player_name": name,
                            "source": "NPB.jp roster",
                            "target_date": target_date,
                        })
            # Hard stop when we reach the next date block.
            if tag in {"h2","h3"} and node is not root:
                title = re.sub(r"\s+", " ", "".join(node.itertext())).strip()
                if "日の出場選手登録" in title and marker not in title:
                    break

    return {
        "source":"NPB.jp roster",
        "target_date":target_date,
        "available":bool(root is not None),
        "text":snippet[:5000],
        "events":events,
    }


def fetch_lineup(game_id: str, target_date: str, home: str, away: str):
    """Fetch a current lineup using the explicit game date and side metadata."""
    if not game_id:
        return {"home": [], "away": []}
    attempts = [
        {"gameId":game_id, "matchDate":target_date},
        {"GameID":game_id, "MatchDate":target_date},
        {"gameId":game_id},
        {"GameID":game_id},
    ]
    raw = None
    for params in attempts:
        try:
            raw = get(f"{SPAIA}/starting_members_for_flash", params).json()
            if raw not in (None,{},[]):
                break
        except Exception:
            raw = None
    out = {"home":[], "away":[]}
    if raw is None:
        return out

    def walk(x):
        if isinstance(x, dict):
            yield x
            for v in x.values():
                yield from walk(v)
        elif isinstance(x, list):
            for v in x:
                yield from walk(v)

    keys_id=("PlayerCD","PlayerId","playerId","playerCD","BatterCD","player_id")
    keys_name=("PlayerName","playerName","BatterName","Name","name","選手名")
    keys_order=("BattingOrder","battingOrder","Order","order","打順")
    keys_side=("side","Side","team","Team","teamName","TeamName","HomeAway","homeAway")

    for d in walk(raw):
        pid=next((d.get(k) for k in keys_id if d.get(k) not in (None,"","-")),None)
        if pid is None:
            continue
        name=next((d.get(k) for k in keys_name if d.get(k) not in (None,"","-")), "")
        order=next((d.get(k) for k in keys_order if d.get(k) not in (None,"","-")), None)
        raw_side=next((d.get(k) for k in keys_side if d.get(k) not in (None,"","-")), "")
        blob=" ".join(str(v) for v in d.values())
        side=None
        token=str(raw_side).lower()
        if token in {"home","h","1","home_team","home-team","ホーム"}:
            side="home"
        elif token in {"away","a","2","away_team","away-team","visitor","ビジター"}:
            side="away"
        elif home in blob:
            side="home"
        elif away in blob:
            side="away"
        if side is None:
            continue
        try:
            batting_order=float(order)
        except Exception:
            batting_order=None
        row={"player_id":str(pid),"player_name":str(name),"batting_order":batting_order}
        if not any(x["player_id"]==row["player_id"] for x in out[side]):
            out[side].append(row)

    for side in out:
        out[side]=sorted(
            out[side],
            key=lambda x:(999 if x["batting_order"] is None else x["batting_order"],x["player_id"])
        )[:12]
    return out


def fetch_weather(game):
    if game["venue"] not in PARKS:
        return {"state":"UNKNOWN","source":"Open-Meteo","reason":"venue coordinates unavailable"}
    lat,lon=PARKS[game["venue"]]
    target = f'{game["date"]}T{game["hour"]:02d}:{game["minute"]:02d}'
    data = get(WEATHER, {
        "latitude":lat,"longitude":lon,"hourly":"temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,wind_direction_10m",
        "forecast_days":2,"timezone":"Asia/Tokyo"
    }).json()
    times=data.get("hourly",{}).get("time",[])
    if target not in times:
        # Select closest hour, but record that the requested exact point was not available.
        ts=pd.to_datetime(times,errors="coerce")
        if len(ts)==0:
            return {"state":"UNKNOWN","source":"Open-Meteo","reason":"no hourly forecast"}
        desired=pd.Timestamp(target)
        i=int(np.argmin(np.abs(ts-desired)))
        quality="NEAREST_HOUR"
    else:
        i=times.index(target); quality="EXACT_HOUR"
    h=data["hourly"]
    return {
        "state":"VERIFIED","source":"Open-Meteo forecast","quality":quality,
        "retrieved_at":datetime.now(timezone.utc).isoformat(),
        "valid_time_local":times[i],
        "temperature_c":float(h["temperature_2m"][i]),
        "humidity_pct":float(h["relative_humidity_2m"][i]),
        "precip_mm":float(h["precipitation"][i]),
        "wind_kmh":float(h["wind_speed_10m"][i]),
        "wind_direction_deg":float(h["wind_direction_10m"][i]),
        "available_at":datetime.now(timezone.utc).isoformat(),
    }


def rest_travel(historical, game):
    if historical.empty:
        return {"state":"UNKNOWN"}
    result={}
    for side, team in (("home",game["home"]),("away",game["away"])):
        sub=historical[(historical["home"]==team)|(historical["away"]==team)].sort_values("datetime")
        if sub.empty:
            result[side]={"rest_days":30.0,"games_last_3d":0,"games_last_7d":0,"travel_miles":0.0}
            continue
        last=sub.iloc[-1]
        dt=pd.Timestamp(game["datetime"])
        last_dt=pd.Timestamp(last["datetime"])
        rest=max(0.0,(dt-last_dt).total_seconds()/86400)
        prev_venue=str(last.get("venue",""))
        cur_venue=str(game.get("venue",""))
        miles=0.0
        if prev_venue in PARKS and cur_venue in PARKS:
            lat1,lon1=PARKS[prev_venue]; lat2,lon2=PARKS[cur_venue]
            # Haversine.
            p=np.pi/180.0
            a=np.sin((lat2-lat1)*p/2)**2+np.cos(lat1*p)*np.cos(lat2*p)*np.sin((lon2-lon1)*p/2)**2
            miles=3958.7613*2*np.arcsin(np.sqrt(a))
        result[side]={"rest_days":float(rest),"games_last_3d":int(((sub["datetime"]>=last_dt-pd.Timedelta(days=3)) & (sub["datetime"]<dt)).sum()),"games_last_7d":int(((sub["datetime"]>=last_dt-pd.Timedelta(days=7)) & (sub["datetime"]<dt)).sum()),"travel_miles":float(miles)}
    result["state"]="VERIFIED"
    return result


def build_matchday_observations(game: dict, prediction_time: str) -> list[dict]:
    """Convert current context into a persistent PIT observation ledger."""
    retrieved = prediction_time
    gid = str(game.get("game_id") or "")
    rows: list[dict] = []

    def freshness(available_at):
        try:
            delta = (
                pd.to_datetime(prediction_time, utc=True)
                - pd.to_datetime(available_at, utc=True)
            ).total_seconds()
            return float(max(0.0, delta))
        except Exception:
            return None

    def add(kind, state, source, value, available_at, source_time=None, confidence=1.0):
        if not available_at:
            return
        rows.append({
            "game_id": gid,
            "snapshot_id": stable_hash({
                "game_id": gid,
                "prediction_time": prediction_time,
                "kind": kind,
                "value": value,
            })[:16],
            "prediction_time": prediction_time,
            "available_at": str(available_at),
            "source_time": str(source_time or available_at),
            "retrieved_at": retrieved,
            "kind": str(kind),
            "state": str(state),
            "source": str(source),
            "value": value,
            "freshness_seconds": freshness(available_at),
            "confidence": float(np.clip(confidence, 0.0, 1.0)),
        })

    starter_state = game.get("starter_state", "UNKNOWN")
    starter_available = game.get("starter_available_at")
    add(
        ContextKind.STARTER.value,
        starter_state,
        game.get("starter_source", "NPB.jp"),
        {"home": game.get("starter_home", ""), "away": game.get("starter_away", "")},
        starter_available,
    )
    if starter_state in {"VERIFIED", "PROJECTED"} and starter_available:
        add(
            ContextKind.STARTER.value,
            starter_state,
            game.get("starter_source", "NPB.jp"),
            "STARTER_CONFIRMED",
            starter_available,
            confidence=1.0 if starter_state == "VERIFIED" else 0.7,
        )

    lineup_state = game.get("lineup_state", "UNKNOWN")
    lineup_available = game.get("lineup_available_at")
    add(
        ContextKind.LINEUP.value,
        lineup_state,
        game.get("lineup_source", "SPAIA"),
        {"home": game.get("lineup", {}).get("home", []), "away": game.get("lineup", {}).get("away", [])},
        lineup_available,
    )
    if lineup_state in {"VERIFIED", "PROJECTED"} and lineup_available:
        add(
            ContextKind.LINEUP.value,
            lineup_state,
            game.get("lineup_source", "SPAIA"),
            "LINEUP_CONFIRMED" if lineup_state == "VERIFIED" else "LINEUP_PROJECTED",
            lineup_available,
            confidence=1.0 if lineup_state == "VERIFIED" else 0.7,
        )

    w = game.get("weather") or {}
    if w.get("state") not in (None, "", "UNKNOWN"):
        add(
            ContextKind.WEATHER.value,
            game.get("weather_state", w.get("state", "UNKNOWN")),
            w.get("source", "Open-Meteo"),
            {
                "temperature_c": w.get("temperature_c"),
                "humidity_pct": w.get("humidity_pct"),
                "precip_mm": w.get("precip_mm"),
                "wind_kmh": w.get("wind_kmh"),
                "wind_direction_deg": w.get("wind_direction_deg"),
            },
            game.get("weather_available_at") or w.get("available_at"),
        )

    for ev in (game.get("roster_events") or []):
        event_name = "PLAYER_OUT" if ev.get("status") == "REMOVED" else "PLAYER_RETURNED"
        add(
            ContextKind.AVAILABILITY.value,
            "VERIFIED",
            ev.get("source", "NPB.jp roster"),
            event_name,
            prediction_time,
        )

    rest = game.get("rest_travel") or {}
    if rest.get("state") == "VERIFIED":
        add(
            ContextKind.REST_TRAVEL.value,
            "VERIFIED",
            "derived from completed historical schedule",
            rest,
            prediction_time,
            confidence=0.8,
        )
    return rows


def persist_matchday_forward_ledger(predictions: list[dict]) -> None:
    """Append prediction/context snapshots for future settlement and replay."""
    snapshot_path = RESULTS / "matchday_snapshots.jsonl"
    baseline_path = RESULTS / "matchday_baseline.csv"
    snapshot_rows = []
    baseline_rows = []

    for game in predictions:
        pt = str(game.get("prediction_time_utc") or "")
        gid = str(game.get("game_id") or "")
        if not gid or not pt:
            continue
        obs = build_matchday_observations(game, pt)
        snapshot_rows.append(json.dumps({
            "game_id": gid,
            "prediction_time_utc": pt,
            "observations": obs,
        }, ensure_ascii=False, sort_keys=True))

        pred = game.get("prediction") or {}
        if pred.get("incumbent_status") == "PASS":
            baseline_rows.append({
                "game_id": gid,
                "datetime": game.get("datetime"),
                "prediction_time_utc": pt,
                "pred_home": pred.get("incumbent_home"),
                "pred_draw": pred.get("incumbent_draw"),
                "pred_away": pred.get("incumbent_away"),
                "pred_shadow_home": pred.get("shadow_home"),
                "pred_shadow_draw": pred.get("shadow_draw"),
                "pred_shadow_away": pred.get("shadow_away"),
                "shadow_status": pred.get("shadow_status"),
                "actual": np.nan,
            })

    snapshot_path.touch(exist_ok=True)
    if snapshot_rows:
        with snapshot_path.open("a", encoding="utf-8") as fh:
            for row in snapshot_rows:
                fh.write(row + "\n")

    if baseline_rows:
        bdf = pd.DataFrame(baseline_rows)
        if baseline_path.exists() and baseline_path.stat().st_size:
            try:
                old = pd.read_csv(baseline_path)
            except Exception:
                old = pd.DataFrame()
            bdf = pd.concat([old, bdf], ignore_index=True)
        bdf.drop_duplicates(["game_id", "prediction_time_utc"], keep="last").to_csv(
            baseline_path, index=False
        )
    elif not baseline_path.exists():
        pd.DataFrame(columns=[
            "game_id","datetime","prediction_time_utc","pred_home","pred_draw",
            "pred_away","pred_shadow_home","pred_shadow_draw","pred_shadow_away",
            "shadow_status","actual"
        ]).to_csv(baseline_path, index=False)


def stable_hash(obj):
    return hashlib.sha256(json.dumps(obj,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()).hexdigest()


def main() -> int:
    now=datetime.now(JST)
    target_date=now.strftime("%Y-%m-%d")
    year=now.year
    RESULTS.mkdir(parents=True,exist_ok=True)

    # Context collection is independent from model availability: a temporary
    # historical-data failure must not erase the current matchday snapshot.
    try:
        games=fetch_schedule(year,target_date)
    except Exception as exc:
        games=[]
        schedule_error=f"{type(exc).__name__}: {exc}"
    else:
        schedule_error=""

    try:
        gid_map=fetch_spaia_game_ids(year,target_date)
    except Exception as exc:
        gid_map={}
        gid_error=f"{type(exc).__name__}: {exc}"
    else:
        gid_error=""

    try:
        starters=fetch_official_starters(target_date)
        starter_error=""
    except Exception as exc:
        starters={}
        starter_error=f"{type(exc).__name__}: {exc}"

    try:
        roster=fetch_roster_notice(target_date)
    except Exception as exc:
        roster={"source":"NPB.jp roster","target_date":target_date,"available":False,"text":"","error":f"{type(exc).__name__}: {exc}"}

    # Best-effort historical model preparation.
    bt=BaseballBacktest(DATA)
    historical=pd.DataFrame()
    X_hist=pd.DataFrame()
    fitted=None
    model_error=""
    checkpoints=DATA/"checkpoints"/"npb_walkforward.csv"
    routing_artifact=RESULTS/"routing_oos_replay.json"

    try:
        historical_raw=bt.load_npb_pbp()
        historical=bt.aggregate_npb_games(historical_raw)
        historical["datetime"]=pd.to_datetime(historical["datetime"],errors="coerce",utc=True)
        cutoff=pd.Timestamp(now).tz_convert("UTC")
        historical=historical[historical["datetime"]<cutoff].copy()
        X_hist,y_hist,_=bt.build_features(historical.assign(league="NPB"))
        if len(X_hist)>=150:
            fitted,_,_=bt.fit_ensemble(X_hist,y_hist,"NPB")
        else:
            model_error=f"historical rows below model minimum: {len(X_hist)}"
    except Exception as exc:
        model_error=f"{type(exc).__name__}: {exc}"

    predictions=[]
    games = [
        g for g in games
        if pd.Timestamp(
            f'{g["date"]} {g["hour"]:02d}:{g["minute"]:02d}', tz="Asia/Tokyo"
        ) > now
    ]
    for g in games:
        g["game_id"]=gid_map.get((g["home"],g["away"]),"")
        local_dt=pd.Timestamp(f'{g["date"]} {g["hour"]:02d}:{g["minute"]:02d}',tz="Asia/Tokyo")
        g["datetime"]=local_dt.tz_convert("UTC")
        g["starter_home"]=starters.get(g["home"],"")
        g["starter_away"]=starters.get(g["away"],"")
        g["starter_state"]="VERIFIED" if g["starter_home"] and g["starter_away"] else "UNKNOWN"
        g["starter_source"]="NPB.jp official announced starters" if g["starter_state"]=="VERIFIED" else ""
        g["starter_available_at"]=now.astimezone(timezone.utc).isoformat() if g["starter_state"]=="VERIFIED" else ""
        g["prediction_time_utc"]=now.astimezone(timezone.utc).isoformat()
        try:
            g["lineup"]=fetch_lineup(g["game_id"],target_date,g["home"],g["away"]) if g["game_id"] else {"home":[],"away":[]}
        except Exception:
            g["lineup"]={"home":[],"away":[]}
        g["lineup_state"]="VERIFIED" if g["lineup"]["home"] and g["lineup"]["away"] else "UNKNOWN"
        g["lineup_source"]="SPAIA starting_members_for_flash" if g["lineup_state"]=="VERIFIED" else ""
        g["lineup_available_at"]=now.astimezone(timezone.utc).isoformat() if g["lineup_state"]=="VERIFIED" else ""
        try:
            w=fetch_weather(g)
        except Exception as exc:
            w={"state":"UNKNOWN","source":"Open-Meteo","reason":f"{type(exc).__name__}: {exc}"}
        g["weather"]=w
        if w.get("state")=="VERIFIED":
            g["weather_available_at"]=w.get("available_at","")
            g["weather_state"]="PROJECTED"
        else:
            g["weather_available_at"]=""; g["weather_state"]="UNKNOWN"
        roster_events=[]
        for ev in roster.get("events", []):
            if ev.get("team") in {g["home"], g["away"]}:
                roster_events.append(ev)
        g["roster_events"]=roster_events
        g["rest_travel"]=rest_travel(historical,g)

        pred_payload={"incumbent_status":"DEFERRED","shadow_status":"DEFERRED","shadow_not_promoted":True}
        if fitted is not None:
            row=pd.Series({
                "league":"NPB","game_id":g["game_id"],"datetime":g["datetime"],"home":g["home"],"away":g["away"],
                "home_starter":g["starter_home"],"away_starter":g["starter_away"],
                "home_lineup_json":json.dumps(g["lineup"]["home"],ensure_ascii=False),
                "away_lineup_json":json.dumps(g["lineup"]["away"],ensure_ascii=False),
                "prediction_time_utc":g["prediction_time_utc"],
                "home_lineup_available_at":g["lineup_available_at"],"away_lineup_available_at":g["lineup_available_at"],
                "home_lineup_state":g["lineup_state"],"away_lineup_state":g["lineup_state"],
                "lineup_available_at":g["lineup_available_at"],"lineup_state":g["lineup_state"],
                "weather_available_at":g["weather_available_at"],"weather_state":g["weather_state"],
                "weather_temp_c":w.get("temperature_c",0.0),"weather_humidity_pct":w.get("humidity_pct",0.0),
                "weather_wind_kmh":w.get("wind_kmh",0.0),"weather_precip_mm":w.get("precip_mm",0.0),
            })
            try:
                fx=pd.DataFrame([bt.match_features(row)])
                incumbent=bt.ensemble_proba(fitted,fx)[0]
                pred_payload.update({
                    "incumbent_status":"PASS",
                    "incumbent_home":float(incumbent[0]),
                    "incumbent_draw":float(incumbent[1]),
                    "incumbent_away":float(incumbent[2]),
                    "incumbent_model":"Ensemble(" + "+".join(x[2] for x in fitted) + ")",
                })
                if checkpoints.exists():
                    try:
                        ck_u=pd.read_csv(checkpoints).sort_values(["datetime","game_id"]).tail(160)
                        if {"actual","pred_home","pred_draw","pred_away"}.issubset(ck_u.columns) and len(ck_u)>=45:
                            hp=ck_u[["pred_home","pred_draw","pred_away"]].to_numpy(dtype=float)
                            hy=ck_u["actual"].to_numpy(dtype=int)
                            if np.all(np.isfinite(hp)) and np.all(np.isfinite(hy)):
                                pred_payload["conformal_uncertainty"]=uncertainty_summary(
                                    incumbent,hp,hy,alpha=0.10
                                )
                                cu=pred_payload["conformal_uncertainty"]
                                k_classes=len(incumbent)
                                set_risk=float(np.clip(
                                    (int(cu.get("prediction_set_size", 1))-1)/max(k_classes-1,1),
                                    0.0,1.0
                                ))
                                gap_risk=float(np.clip(
                                    1.0-float(cu.get("top_gap",0.0))/0.25,
                                    0.0,1.0
                                ))
                                pred_payload["conformal_signal"]=float(np.clip(
                                    0.6*set_risk+0.4*gap_risk,0.0,1.0
                                ))
                    except Exception:
                        pass

                if checkpoints.exists():
                    ck=pd.read_csv(checkpoints)
                    keys=sorted({c[len("expert_"):].rsplit("_",1)[0] for c in ck.columns if c.startswith("expert_") and c.endswith("_home")})
                    current_raw=[]; current_names=[]
                    for model,_w,name in fitted:
                        key="".join(ch.lower() if ch.isalnum() else "_" for ch in str(name)).strip("_")
                        if key not in keys: continue
                        current_raw.append(bt.align_proba(model.predict_proba(fx),model.classes_,"NPB")[0])
                        current_names.append(key)

                    if len(current_raw)>=2:
                        hist_losses=[]; hist_probs=[]; hist_y=[]
                        ordered_ck=ck.sort_values(["datetime","game_id"])
                        for _,rrw in ordered_ck.tail(1200).iterrows():
                            try:
                                yy=int(rrw["actual"]); probs=[]
                                for key in current_names:
                                    cols=[f"expert_{key}_home",f"expert_{key}_draw",f"expert_{key}_away"]
                                    pp=np.asarray([float(rrw[n]) for n in cols],dtype=float)
                                    if not np.all(np.isfinite(pp)): raise ValueError("non-finite probability")
                                    pp=np.clip(pp,1e-12,1.0); pp/=pp.sum(); probs.append(pp)
                                hist_probs.append(probs);hist_y.append(yy)
                            except Exception:
                                continue

                        if len(hist_probs)>=45:
                            raw=np.asarray(hist_probs[-120:],dtype=float); ys=np.asarray(hist_y[-120:],dtype=int)
                            bank=ExpertCalibrationBank(len(current_raw),config=RoutingConfig())
                            bank.update(raw,ys)
                            cal_current=bank.predict(np.asarray(current_raw))
                            cal_hist=np.asarray([bank.predict(z) for z in raw],dtype=float)
                            loss_hist=-np.log(np.clip(
                                np.take_along_axis(cal_hist,ys.reshape(-1,1)[:,None],axis=2).squeeze(-1),
                                1e-12,1.0,
                            ))
                            # The expression above is intentionally replaced below
                            # with a shape-safe explicit loop for clarity.
                            loss_hist=np.empty((len(cal_hist),len(current_raw)),dtype=float)
                            for hi,z in enumerate(cal_hist):
                                loss_hist[hi,:]=-np.log(np.clip(z[:,ys[hi]],1e-12,1.0))

                            fitted_weight_map={name:float(wt) for _m,wt,name in fitted}
                            prev=np.asarray([fitted_weight_map.get(name,0.0) for name in current_names],dtype=float)
                            if prev.sum()<=0: prev=None

                            output_drift=0.0
                            if len(hist_probs)>=24:
                                arr=np.asarray(hist_probs[-96:],dtype=float)
                                short=arr[-12:].mean(axis=0)
                                old=arr[:-12]
                                if len(old)>=12:
                                    output_drift=float(np.clip(1.0-np.exp(-float(np.mean(np.abs(short-old.mean(axis=0))))/0.12),0,1))
                            feature_drift=0.0
                            try:
                                if len(X_hist)>=24:
                                    recent=X_hist.tail(12)
                                    old=X_hist.iloc[max(0,len(X_hist)-96):-12]
                                    common=[c for c in recent.columns if c in fx.columns]
                                    if len(common)>=8 and len(old)>=12:
                                        feature_drift=feature_drift_score(old[common].to_numpy(dtype=float),recent[common].to_numpy(dtype=float))
                            except Exception:
                                feature_drift=0.0
                            drift=float(np.clip(0.70*output_drift+0.30*feature_drift,0,1))
                            rr=route_experts(
                                cal_current,
                                loss_hist,
                                drift_score=drift,
                                previous_weights=prev,
                                conformal_uncertainty=float(pred_payload.get("conformal_signal",0.0)),
                            )
                            mixed=mix_expert_probabilities(cal_current,rr.weights)

                            # Canonical Matchday reforecast is invoked only when the
                            # OOS-learned context effect artifact is eligible. Otherwise
                            # the routed/recalibrated baseline remains unchanged.
                            final = mixed.copy()
                            matchday_result = None
                            effect_artifact = RESULTS / "matchday_effects.json"
                            try:
                                observations = [
                                    Observation(**x)
                                    for x in build_matchday_observations(
                                        g, g["prediction_time_utc"]
                                    )
                                ]
                                if effect_artifact.exists():
                                    cal_temp = 1.0
                                    gate_file = RESULTS / "matchday_integrated_acceptance_gate.json"
                                    if routing_artifact.exists() and gate_file.exists():
                                        gate = json.loads(gate_file.read_text(encoding="utf-8"))
                                        if gate.get("candidate_eligible") is True:
                                            ra = json.loads(routing_artifact.read_text(encoding="utf-8"))
                                            temps = [
                                                float(x.get("final_temperature", 1.0))
                                                for x in ra.get("results", [])
                                                if x.get("status") == "PASS"
                                            ]
                                            if temps:
                                                cal_temp = temps[-1]
                                    final_cal = AdaptiveTemperatureCalibrator(
                                        temperature=cal_temp,
                                        config=RoutingConfig(),
                                    )
                                    matchday_result = reforecast(
                                        baseline=np.asarray(incumbent, dtype=float),
                                        expert_probs=np.asarray(cal_current, dtype=float),
                                        history_logloss=np.asarray(loss_hist, dtype=float),
                                        observations=observations,
                                        previous_weights=prev,
                                        drift_score=drift,
                                        conformal_uncertainty=float(pred_payload.get("conformal_signal",0.0)),
                                        calibrator=final_cal,
                                        config=RoutingConfig(),
                                    )
                                    if matchday_result.status == "PASS":
                                        final = matchday_result.final
                            except Exception as exc:
                                pred_payload["matchday_reforecast_reason"] = (
                                    f"{type(exc).__name__}: {exc}"
                                )
                            pred_payload.update({
                                "shadow_status":"PASS",
                                "shadow_home":float(final[0]),"shadow_draw":float(final[1]),"shadow_away":float(final[2]),
                                "shadow_pre_matchday_home":float(mixed[0]),"shadow_pre_matchday_draw":float(mixed[1]),"shadow_pre_matchday_away":float(mixed[2]),
                                "matchday_reforecast_status":(
                                    matchday_result.status if matchday_result is not None else "DEFERRED"
                                ),
                                "matchday_reforecast_applied_events":(
                                    matchday_result.applied_events if matchday_result is not None else []
                                ),
                                "matchday_reforecast_skipped_events":(
                                    matchday_result.skipped_events if matchday_result is not None else []
                                ),
                                "matchday_reforecast_reason":(
                                    matchday_result.reason if matchday_result is not None else "no integrated acceptance gate"
                                ),
                                "shadow_weights":rr.weights.tolist(),"shadow_temperature":(
                                    getattr(final_cal, "temperature", 1.0) if matchday_result is not None else 1.0
                                ),
                                "shadow_uncertainty":float(rr.uncertainty),"shadow_disagreement":float(rr.disagreement),
                                "shadow_drift":drift,"shadow_feature_drift":feature_drift,"shadow_output_drift":output_drift,
                            })
            except Exception as exc:
                pred_payload["shadow_reason"]=f"{type(exc).__name__}: {exc}"
        if model_error and pred_payload["incumbent_status"]=="DEFERRED":
            pred_payload["incumbent_reason"]=model_error
        g["prediction"]=pred_payload
        predictions.append(g)

    persist_matchday_forward_ledger(predictions)

    payload={
        "schema_version":1,
        "status":"PASS" if games or not schedule_error else "DEFERRED",
        "prediction_time_utc":now.astimezone(timezone.utc).isoformat(),
        "target_date_jst":target_date,
        "source_policy":{"schedule":"NPB.jp","starter":"NPB.jp announcement","lineup":"SPAIA current snapshot","weather":"Open-Meteo forecast","market":"UNKNOWN/no paid source"},
        "collection_diagnostics":{"schedule_error":schedule_error,"spaia_schedule_error":gid_error,"starter_error":starter_error,"model_error":model_error},
        "roster_notice":roster,
        "games":predictions,
    }
    stable_payload=json.loads(json.dumps(payload,ensure_ascii=False,default=str))
    stable_payload.pop("prediction_time_utc",None)
    for gg in stable_payload.get("games",[]):
        for key in ("prediction_time_utc","starter_available_at","lineup_available_at"):
            gg.pop(key,None)
        w=gg.get("weather")
        if isinstance(w,dict):
            w.pop("retrieved_at",None);w.pop("available_at",None)
    stable_hash_value=stable_hash(stable_payload)
    payload["state_hash"]=stable_hash_value
    target=RESULTS/"matchday_intelligence_current.json"
    old_hash=None
    if target.exists():
        try: old_hash=json.loads(target.read_text(encoding="utf-8")).get("state_hash")
        except Exception: old_hash=None
    if old_hash != stable_hash_value:
        target.write_text(json.dumps(payload,ensure_ascii=False,indent=2,default=str)+"\n",encoding="utf-8")
    print(json.dumps({"status":payload["status"],"games":len(predictions),"state_hash":stable_hash_value,"material_change":old_hash!=stable_hash_value},ensure_ascii=False))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
