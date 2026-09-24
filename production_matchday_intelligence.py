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
    AdaptiveTemperatureCalibrator,
    ExpertCalibrationBank,
    RoutingConfig,
    feature_drift_score,
    mix_expert_probabilities,
    route_experts,
)

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
    raw = get(f"{SPAIA}/schedules", {"Year": year}).json()
    result = {}
    for g in raw if isinstance(raw, list) else []:
        d = pd.to_datetime(g.get("datetime") or g.get("DateJPN") or g.get("date"), errors="coerce")
        if pd.isna(d) or d.strftime("%Y-%m-%d") != target_date:
            continue
        home = norm_team(g.get("HTeamNameS") or g.get("home_team_short_name") or g.get("homeTeam") or g.get("HomeTeamName"))
        away = norm_team(g.get("VTeamNameS") or g.get("away_team_short_name") or g.get("visitorTeam") or g.get("AwayTeamName"))
        gid = str(g.get("GameID") or g.get("gameId") or g.get("game_id") or g.get("gamePk") or "").strip()
        if home and away and gid:
            result[(home,away)] = gid
    return result


def fetch_official_starters():
    r = get(f"{NPB}/announcement/starter/")
    doc = lxml_html.fromstring(r.content)
    names = []
    teams = []
    for img in doc.xpath("//img[@alt]"):
        alt = str(img.get("alt") or "").strip()
        if norm_team(alt) in TEAM_NAMES:
            teams.append((img, norm_team(alt)))
    links = doc.xpath("//a")
    ordered = []
    for a in links:
        txt = re.sub(r"\s+", " ", "".join(a.itertext())).strip()
        href = str(a.get("href") or "")
        if txt and ("/bis/players/" in href or "/player/" in href):
            ordered.append((a,txt))
    starters = {}
    # Team-image order and player-link order are adjacent on the NPB page.
    for img, team in teams:
        element_index = next((i for i,a in enumerate(doc.iterlinks()) if False), None)
        parent = img.getparent()
        if parent is None:
            continue
        for a in parent.xpath("following::a"):
            txt = re.sub(r"\s+", " ", "".join(a.itertext())).strip()
            href = str(a.get("href") or "")
            if txt and ("/bis/players/" in href or "/player/" in href):
                starters[team] = txt
                break
    return starters


def fetch_roster_notice(target_date: str):
    r = get(f"{NPB}/announcement/roster/")
    text = re.sub(r"\s+", " ", lxml_html.fromstring(r.content).text_content())
    marker = f"{int(target_date[5:7])}月{int(target_date[8:10])}日の出場選手登録"
    pos = text.find(marker)
    snippet = text[pos:pos+8000] if pos >= 0 else text[:8000]
    return {"source":"NPB.jp roster","target_date":target_date,"available":bool(pos >= 0),"text":snippet[:5000]}


def fetch_lineup(game_id: str, home: str, away: str):
    attempts = [
        {"gameId":game_id,"matchDate":home},
        {"gameId":game_id},
        {"GameID":game_id},
    ]
    raw = None
    for params in attempts:
        try:
            raw = get(f"{SPAIA}/starting_members_for_flash", params).json()
            if raw not in (None,{},[]): break
        except Exception:
            raw = None
    out = {"home":[],"away":[]}
    if raw is None:
        return out
    def walk(x):
        if isinstance(x,dict):
            yield x
            for v in x.values(): yield from walk(v)
        elif isinstance(x,list):
            for v in x: yield from walk(v)
    keys_id=("PlayerCD","PlayerId","playerId","playerCD","BatterCD","player_id")
    keys_name=("PlayerName","playerName","BatterName","Name","name","選手名")
    keys_order=("BattingOrder","battingOrder","Order","order","打順")
    for d in walk(raw):
        pid=next((d.get(k) for k in keys_id if d.get(k) not in (None,"","-")),None)
        name=next((d.get(k) for k in keys_name if d.get(k) not in (None,"","-")), "")
        order=next((d.get(k) for k in keys_order if d.get(k) not in (None,"","-")), None)
        blob=" ".join(str(v) for v in d.values()).lower()
        side="home" if str(home).lower() in blob else "away" if str(away).lower() in blob else None
        if pid is None or side is None:
            continue
        row={"player_id":str(pid),"player_name":str(name),"batting_order":float(order) if str(order).replace(".","",1).isdigit() else None}
        if not any(x["player_id"]==row["player_id"] for x in out[side]):
            out[side].append(row)
    for side in out:
        out[side]=sorted(out[side],key=lambda x:(999 if x["batting_order"] is None else x["batting_order"],x["player_id"]))[:12]
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


def stable_hash(obj):
    return hashlib.sha256(json.dumps(obj,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()).hexdigest()


def main() -> int:
    now=datetime.now(JST)
    target_date=now.strftime("%Y-%m-%d")
    year=now.year
    RESULTS.mkdir(parents=True,exist_ok=True)

    try:
        games=fetch_schedule(year,target_date)
        gid_map=fetch_spaia_game_ids(year,target_date)
        starters=fetch_official_starters()
        roster=fetch_roster_notice(target_date)

        bt=BaseballBacktest(DATA)
        historical_raw=bt.load_npb_pbp()
        historical=bt.aggregate_npb_games(historical_raw)
        # Only completed games strictly before the current target are valid history.
        historical["datetime"]=pd.to_datetime(historical["datetime"],errors="coerce",utc=True)
        cutoff=pd.Timestamp(now).tz_convert("UTC")
        historical=historical[historical["datetime"]<cutoff].copy()
        X_hist,y_hist,_=bt.build_features(historical.assign(league="NPB"))

        fitted,val_scores,best_name=bt.fit_ensemble(X_hist,y_hist,"NPB") if len(X_hist)>=150 else (None,{},None)
        predictions=[]
        checkpoints=RESULTS/"checkpoints"/"npb_walkforward.csv"
        routing_artifact=RESULTS/"routing_oos_replay.json"
        gate_artifact=RESULTS/"routing_acceptance_gate.json"

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
            g["lineup"]=fetch_lineup(g["game_id"],g["home"],g["away"]) if g["game_id"] else {"home":[],"away":[]}
            g["lineup_state"]="VERIFIED" if g["lineup"]["home"] and g["lineup"]["away"] else "UNKNOWN"
            g["lineup_source"]="SPAIA starting_members_for_flash" if g["lineup_state"]=="VERIFIED" else ""
            g["lineup_available_at"]=now.astimezone(timezone.utc).isoformat() if g["lineup_state"]=="VERIFIED" else ""
            w=fetch_weather(g); g["weather"]=w
            if w.get("state")=="VERIFIED":
                g["weather_available_at"]=w["available_at"]; g["weather_state"]="PROJECTED"
            else:
                g["weather_available_at"]=""; g["weather_state"]="UNKNOWN"
            rt=rest_travel(historical,g); g["rest_travel"]=rt
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
            pred_payload={"incumbent_status":"DEFERRED","shadow_status":"DEFERRED"}
            if fitted is not None:
                fx=pd.DataFrame([bt.match_features(row)])
                incumbent=bt.ensemble_proba(fitted,fx)[0]
                pred_payload.update({
                    "incumbent_status":"PASS",
                    "incumbent_home":float(incumbent[0]),
                    "incumbent_draw":float(incumbent[1]),
                    "incumbent_away":float(incumbent[2]),
                    "incumbent_model":str("Ensemble(" + "+".join(x[2] for x in fitted) + ")"),
                })
                # Research-only dynamic routing shadow.
                if checkpoints.exists():
                    try:
                        ck=pd.read_csv(checkpoints)
                        keys=sorted({c[len("expert_"):].rsplit("_",1)[0] for c in ck.columns if c.startswith("expert_") and c.endswith("_home")})
                        current_raw=[]
                        current_names=[]
                        for model,_w,name in fitted:
                            key="".join(ch.lower() if ch.isalnum() else "_" for ch in str(name)).strip("_")
                            if key not in keys: continue
                            pp=bt.align_proba(model.predict_proba(fx),model.classes_,"NPB")[0]
                            current_raw.append(pp);current_names.append(key)
                        if len(current_raw)>=2:
                            hist_losses=[]
                            hist_probs=[]
                            hist_y=[]
                            ordered_ck=ck.sort_values(["datetime","game_id"])
                            for _,rr in ordered_ck.tail(1200).iterrows():
                                try:
                                    yy=int(rr["actual"])
                                    probs=[]
                                    losses=[]
                                    for key in current_names:
                                        names=[f"expert_{key}_home",f"expert_{key}_draw",f"expert_{key}_away"]
                                        pp=np.asarray([float(rr[n]) for n in names],dtype=float)
                                        if not np.all(np.isfinite(pp)):
                                            raise ValueError("non-finite historical expert probability")
                                        pp=np.clip(pp,1e-12,1.0);pp/=pp.sum()
                                        probs.append(pp);losses.append(float(-np.log(pp[yy])))
                                    hist_probs.append(probs);hist_losses.append(losses);hist_y.append(yy)
                                except Exception:
                                    continue
                            if len(hist_losses)>=45:
                                raw=np.asarray(hist_probs[-120:],dtype=float)
                                ys=np.asarray(hist_y[-120:],dtype=int)
                                bank=ExpertCalibrationBank(len(current_raw),config=RoutingConfig())
                                bank.update(raw,ys)
                                cal_current=bank.predict(np.asarray(current_raw))
                                loss_hist=np.asarray(hist_losses[-120:],dtype=float)
                                # Incumbent ensemble weights are the stable anchor for
                                # high-uncertainty routing; only matched experts are used.
                                fitted_weight_map={name:float(w) for _m,w,name in fitted}
                                prev=np.asarray([fitted_weight_map.get(name,0.0) for name in current_names],dtype=float)
                                if prev.sum() <= 0:
                                    prev=None
                                feature_drift=0.0
                                try:
                                    if len(X_hist) >= 24:
                                        recent_hist=X_hist.tail(12)
                                        old_hist=X_hist.iloc[max(0,len(X_hist)-96):-12]
                                        common=[c for c in recent_hist.columns if c in fx.columns]
                                        if len(common)>=8 and len(old_hist)>=12:
                                            feature_drift=feature_drift_score(
                                                old_hist[common].to_numpy(dtype=float),
                                                recent_hist[common].to_numpy(dtype=float),
                                            )
                                except Exception:
                                    feature_drift=0.0
                                output_drift=0.0
                                if len(hist_probs)>=24:
                                    arr=np.asarray(hist_probs[-96:],dtype=float)
                                    short=arr[-12:].mean(axis=0)
                                    old=arr[:-12]
                                    if len(old)>=12:
                                        output_drift=float(np.clip(
                                            1.0-np.exp(-float(np.mean(np.abs(short-old.mean(axis=0))))/0.12),
                                            0.0,1.0,
                                        ))
                                drift=float(np.clip(0.70*output_drift+0.30*feature_drift,0.0,1.0))
                                rr=route_experts(cal_current,loss_hist,drift_score=drift,previous_weights=prev)
                                mixed=mix_expert_probabilities(cal_current,rr.weights)
                                temp=1.0
                                if routing_artifact.exists():
                                    ra=json.loads(routing_artifact.read_text(encoding="utf-8"))
                                    candidates=[x for x in ra.get("results",[]) if x.get("status")=="PASS"]
                                    if candidates:
                                        temp=float(candidates[-1].get("final_temperature",1.0))
                                final=np.asarray(mixed)
                                if abs(temp-1.0)>1e-9:
                                    final=np.power(np.clip(final,1e-12,1.0),1.0/temp);final/=final.sum()
                                pred_payload.update({
                                    "shadow_status":"PASS",
                                    "shadow_home":float(final[0]),"shadow_draw":float(final[1]),"shadow_away":float(final[2]),
                                    "shadow_weights":rr.weights.tolist(),"shadow_temperature":temp,
                                    "shadow_uncertainty":float(rr.uncertainty),
                                    "shadow_drift":float(rr.drift_score),
                                    "shadow_not_promoted":True,
                                })
                    except Exception as exc:
                        pred_payload["shadow_reason"]=f"routing shadow deferred: {type(exc).__name__}: {exc}"
            g["prediction"]=pred_payload
            predictions.append(g)

        payload={
            "schema_version":1,
            "status":"PASS",
            "prediction_time_utc":now.astimezone(timezone.utc).isoformat(),
            "target_date_jst":target_date,
            "source_policy":{"schedule":"NPB.jp","starter":"NPB.jp announcement","lineup":"SPAIA current snapshot","weather":"Open-Meteo forecast","market":"UNKNOWN/no paid source"},
            "roster_notice":roster,
            "games":predictions,
        }
        stable={k:v for k,v in payload.items() if k!="prediction_time_utc"}
        payload["state_hash"]=stable_hash(stable)
        temp=RESULTS/"matchday_intelligence_current.json"
        temp.write_text(json.dumps(payload,ensure_ascii=False,indent=2,default=str)+"\n",encoding="utf-8")
        print(json.dumps({"status":"PASS","games":len(predictions),"state_hash":payload["state_hash"]},ensure_ascii=False))
        return 0
    except Exception as exc:
        payload={
            "schema_version":1,"status":"DEFERRED","reason":f"{type(exc).__name__}: {exc}",
            "prediction_time_utc":now.astimezone(timezone.utc).isoformat(),"target_date_jst":target_date,
        }
        (RESULTS/"matchday_intelligence_current.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        print(json.dumps(payload,ensure_ascii=False))
        return 0


if __name__=="__main__":
    raise SystemExit(main())
