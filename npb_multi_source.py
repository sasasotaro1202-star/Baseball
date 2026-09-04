#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multi-source NPB data preparation for walk-forward backtesting.

Source specialization
----------------------
1) SPAIA NPB API: game schedule/results, pitch-by-pitch starter detection,
   per-game pitcher lines.  This is the primary granular baseball feed.
2) NPB.jp official site: team batting/pitching/standings pages, used as an
   independent audit/validation source.  Current-season snapshots are NOT fed
   into historical pregame features because that would leak future information.
3) Open-Meteo archive: historical hourly weather at the actual ballpark area.
   Weather is aligned to the scheduled local start hour and is used only for
   games whose observation is in the past.

The output is a compact *_pbp.csv compatible with baseball_backtest.py plus
independent audit files.  No large CSV needs to be committed to GitHub.
"""
from __future__ import annotations
import concurrent.futures as cf
import os, re, time, json
from pathlib import Path
from typing import Any, Optional
import numpy as np
import pandas as pd
import requests

SPAIA = "https://spaia.jp/baseball/npb/api"
NPB = "https://npb.jp"
OPEN_METEO = "https://archive-api.open-meteo.com/v1/archive"
YEAR = int(os.getenv("NPB_YEAR", "2026"))
END_DATE = os.getenv("NPB_END_DATE", "2026-09-03")
WORKERS = int(os.getenv("NPB_DOWNLOAD_WORKERS", "6"))
TIMEOUT = 30
OUT = Path("data/2026_multi_source_pbp.csv")

ALIASES = {
"巨人":"読売ジャイアンツ","読売":"読売ジャイアンツ","読売ジャイアンツ":"読売ジャイアンツ",
"阪神":"阪神タイガース","阪神タイガース":"阪神タイガース",
"中日":"中日ドラゴンズ","中日ドラゴンズ":"中日ドラゴンズ",
"広島":"広島東洋カープ","広島東洋":"広島東洋カープ","広島東洋カープ":"広島東洋カープ",
"ヤクルト":"東京ヤクルトスワローズ","東京ヤクルト":"東京ヤクルトスワローズ","東京ヤクルトスワローズ":"東京ヤクルトスワローズ",
"DeNA":"横浜DeNAベイスターズ","ＤｅＮＡ":"横浜DeNAベイスターズ","横浜DeNA":"横浜DeNAベイスターズ","横浜DeNAベイスターズ":"横浜DeNAベイスターズ",
"ソフトバンク":"福岡ソフトバンクホークス","福岡ソフトバンク":"福岡ソフトバンクホークス","福岡ソフトバンクホークス":"福岡ソフトバンクホークス",
"西武":"埼玉西武ライオンズ","埼玉西武":"埼玉西武ライオンズ","埼玉西武ライオンズ":"埼玉西武ライオンズ",
"日本ハム":"北海道日本ハムファイターズ","日ハム":"北海道日本ハムファイターズ","北海道日本ハム":"北海道日本ハムファイターズ","北海道日本ハムファイターズ":"北海道日本ハムファイターズ",
"ロッテ":"千葉ロッテマリーンズ","千葉ロッテ":"千葉ロッテマリーンズ","千葉ロッテマリーンズ":"千葉ロッテマリーンズ",
"楽天":"東北楽天ゴールデンイーグルス","東北楽天":"東北楽天ゴールデンイーグルス","東北楽天ゴールデンイーグルス":"東北楽天ゴールデンイーグルス",
"オリックス":"オリックス・バファローズ","オリックス・バファローズ":"オリックス・バファローズ",
}
# Ballpark coordinates, used only to query historical weather.
PARKS = {
"神　宮":(35.6827,139.6841),"神宮":(35.6827,139.6841),"東京ドーム":(35.7056,139.7519),
"横　浜":(35.4431,139.6400),"横浜":(35.4431,139.6400),"バンテリンドーム":(35.1859,136.9470),
"マツダスタジアム":(34.3916,132.4848),"甲子園":(34.7214,135.3616),"エスコンＦ":(43.0151,141.4094),
"ベルーナドーム":(35.7684,139.4745),"ZOZOマリン":(35.6456,140.0307),"楽天モバイル":(38.2560,140.9014),
"京セラD大阪":(34.6694,135.4761),"ほっと神戸":(34.6795,135.0980),"みずほPayPay":(33.5950,130.3620),
}

def get_json(url, params=None, retries=4):
    last=None
    for i in range(retries):
        try:
            r=requests.get(url,params=params,timeout=TIMEOUT,headers={"User-Agent":"Mozilla/5.0 baseball-backtest"})
            r.raise_for_status(); return r.json()
        except Exception as e:
            last=e; time.sleep(min(1.5*(i+1),5))
    raise RuntimeError(f"request failed: {url}: {last}")

def official_name(x): return ALIASES.get(str(x or '').strip(),str(x or '').strip())

def parse_dt(g):
    d=str(g.get('DateJPN') or g.get('date_jpn') or '')
    t=str(g.get('TimeJPN') or g.get('time_jpn') or '1800')
    try: return pd.to_datetime(d+t,format='%Y%m%d%H%M')
    except Exception: return pd.NaT

def game_kind(g): return str(g.get('GameKindName') or g.get('game_kind_name') or '')

def official_game(g):
    s=game_kind(g)
    if any(x in s for x in ('オープン戦','オールスター','ファーム','二軍','教育')): return False
    return ('公式戦' in s) or ('交流戦' in s) or s==''

def first_pitchers(game_id):
    raw=get_json(f'{SPAIA}/flash_atbat_history',{'gameId':game_id})
    if not isinstance(raw,list): return '',''
    arr=[]
    for x in raw:
        pid=x.get('pitId',x.get('pitcher',x.get('PitcherCD','')))
        ser=str(x.get('fiveDigitSerialNumber',''))
        if pid in (None,'',0) or not ser: continue
        arr.append((ser,str(pid)))
    arr.sort(key=lambda z:z[0])
    away=home=''; seen=set()
    for ser,pid in arr:
        # Common SPAIA serial convention; fallback to first two distinct pitchers.
        if len(ser)>=3:
            half=ser[2]; inning=ser[:2]; key=(inning,half)
            if key in seen: continue
            seen.add(key)
            if half in ('T','t','1') and not away: away=pid
            elif half in ('B','b','2') and not home: home=pid
        if away and home: break
    if not (away and home):
        ids=[]
        for _,pid in arr:
            if pid not in ids: ids.append(pid)
        if len(ids)>=2: away,home=ids[0],ids[1]
    return away,home

def flat_dicts(x):
    if isinstance(x,list):
        out=[]
        for y in x: out += flat_dicts(y)
        return out
    if isinstance(x,dict):
        # Most SPAIA responses contain one or more lists nested under a key.
        direct=[x]
        for v in x.values():
            if isinstance(v,(list,dict)): direct += flat_dicts(v)
        return direct
    return []

def val(d, keys, default=np.nan):
    for k in keys:
        if k in d and d[k] not in (None,'','-'): return d[k]
    return default

def num(v):
    try:
        s=str(v).replace(',','').strip()
        if s in ('','-','nan','None'): return np.nan
        return float(s)
    except: return np.nan

def pitcher_line(game_id, match_date, starter_id):
    try:
        raw=get_json(f'{SPAIA}/both_pitcher_game_stats',{'gameId':game_id,'matchDate':match_date})
    except Exception:
        return None
    candidates=[]
    for d in flat_dicts(raw):
        pid=val(d,['PitcherCD','PlayerCD','PersonInfoID','pitcherId','pitcherID','PitcherId','PlayerId','playerId'],None)
        if pid is not None: candidates.append((str(pid),d))
    exact=[d for pid,d in candidates if str(pid)==str(starter_id)]
    if not exact and candidates: exact=[candidates[0][1]]
    if not exact: return None
    d=exact[0]
    ip=num(val(d,['InningsPitched','IP','inningsPitched','投球回']))
    ip3=num(val(d,['InningsPitched3rd','IP3','inningsPitched3rd']))
    if np.isfinite(ip) and np.isfinite(ip3) and ip3 < 3: ip=ip+ip3/3.0
    h=num(val(d,['HitsAllowed','Hit','Hits','H','被安打']))
    er=num(val(d,['EarnedRun','EarnedRuns','ER','自責点']))
    hr=num(val(d,['HomeRun','HomeRunsAllowed','HR','被本塁打']))
    bb=num(val(d,['Walk','Walks','BB','BaseOnBalls','与四球']))
    so=num(val(d,['Strikeout','Strikeouts','SO','奪三振']))
    if not np.isfinite(ip) or ip<=0: return None
    era=9*er/ip if np.isfinite(er) else np.nan
    whip=(h+bb)/ip if np.isfinite(h) and np.isfinite(bb) else np.nan
    k9=9*so/ip if np.isfinite(so) else np.nan
    bb9=9*bb/ip if np.isfinite(bb) else np.nan
    hr9=9*hr/ip if np.isfinite(hr) else np.nan
    # FIP constant is not known from the game feed; use a neutral run-environment
    # constant only for relative history, never claim it is official FIP.
    fip=(13*hr+3*bb-2*so)/ip+3.20 if all(np.isfinite(x) for x in (hr,bb,so)) else np.nan
    return {'era':era,'whip':whip,'k9':k9,'bb9':bb9,'hr9':hr9,'fip':fip,'ip':ip,'er':er,'h':h,'hr':hr,'bb':bb,'so':so}

def fetch_games():
    raw=get_json(f'{SPAIA}/schedules',{'Year':YEAR})
    end=pd.Timestamp(END_DATE)+pd.Timedelta(days=1)-pd.Timedelta(seconds=1)
    rows=[]
    for g in raw if isinstance(raw,list) else []:
        dt=parse_dt(g)
        if pd.isna(dt) or dt>end or dt<pd.Timestamp(f'{YEAR}-03-20') or not official_game(g): continue
        gid=str(g.get('GameID') or g.get('game_id') or '')
        if not gid: continue
        hs=num(g.get('HScore',g.get('home_score'))); aas=num(g.get('VScore',g.get('away_score')))
        if not np.isfinite(hs) or not np.isfinite(aas): continue
        park=str(g.get('StadiumName') or g.get('stadiumName') or g.get('BallparkName') or g.get('ballpark') or '')
        rows.append({'game_id':gid,'datetime':dt,'home':official_name(g.get('HTeamNameS',g.get('home_team_short_name'))),'away':official_name(g.get('VTeamNameS',g.get('away_team_short_name'))),'home_score':hs,'away_score':aas,'game_type':game_kind(g) or '公式戦','venue':park})
    d=pd.DataFrame(rows).drop_duplicates('game_id').sort_values(['datetime','game_id']).reset_index(drop=True)
    return d

def fetch_game_enrichment(r):
    away,home=first_pitchers(r.game_id)
    date=pd.Timestamp(r.datetime).strftime('%Y%m%d')
    hm=pitcher_line(r.game_id,date,home) if home else None
    am=pitcher_line(r.game_id,date,away) if away else None
    z=r.to_dict(); z.update({'home_starter':home,'away_starter':away})
    for side,m in [('home',hm),('away',am)]:
        for k in ('era','whip','k9','bb9','hr9','fip'):
            z[f'{side}_starter_{k}']=m.get(k) if m else np.nan
    return z

def weather_for_venue(venue, start, end):
    if venue not in PARKS: return pd.DataFrame()
    lat,lon=PARKS[venue]
    try:
        raw=get_json(OPEN_METEO,{'latitude':lat,'longitude':lon,'start_date':start,'end_date':end,'hourly':'temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m','timezone':'Asia/Tokyo'})
    except Exception as e:
        print('[WEATHER SKIP]',venue,e); return pd.DataFrame()
    h=raw.get('hourly',{})
    return pd.DataFrame({'datetime':pd.to_datetime(h.get('time',[])),'weather_temp_c':h.get('temperature_2m',[]),'weather_humidity_pct':h.get('relative_humidity_2m',[]),'weather_precip_mm':h.get('precipitation',[]),'weather_wind_kmh':h.get('wind_speed_10m',[])})

def official_audit():
    out=[]
    for url in [f'{NPB}/bis/{YEAR}/stats/std_c.html',f'{NPB}/bis/{YEAR}/stats/std_p.html',f'{NPB}/bis/{YEAR}/stats/tmb_c.html',f'{NPB}/bis/{YEAR}/stats/tmb_p.html',f'{NPB}/bis/{YEAR}/stats/tmp_c.html',f'{NPB}/bis/{YEAR}/stats/tmp_p.html']:
        try:
            html=requests.get(url,timeout=TIMEOUT,headers={'User-Agent':'Mozilla/5.0'}); html.raise_for_status()
            tables=pd.read_html(html.text)
            for i,t in enumerate(tables):
                if len(t)>0:
                    p=Path('data')/('official_'+url.rsplit('/',1)[-1].replace('.html','')+f'_{i}.csv')
                    t.to_csv(p,index=False)
                    out.append({'url':url,'table_index':i,'rows':len(t),'cols':len(t.columns)})
        except Exception as e: print('[OFFICIAL AUDIT SKIP]',url,e)
    Path('data').mkdir(exist_ok=True)
    pd.DataFrame(out).to_csv('data/source_official_npb_audit.csv',index=False)

def main():
    Path('data').mkdir(exist_ok=True)
    games=fetch_games(); print('[SPAIA] games=',len(games))
    if len(games)<600: raise RuntimeError(f'too few official games: {len(games)}')
    enriched=[]
    with cf.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        fs=[ex.submit(fetch_game_enrichment,r) for r in games.itertuples(index=False)]
        for i,f in enumerate(cf.as_completed(fs),1):
            enriched.append(f.result())
            if i%50==0 or i==len(fs): print(f'[SPAIA] game enrichment {i}/{len(fs)}')
    d=pd.DataFrame(enriched).sort_values(['datetime','game_id']).reset_index(drop=True)
    # Weather: one request per known venue, then exact hour merge.
    for v in sorted(set(d.venue.astype(str))):
        w=weather_for_venue(v,f'{YEAR}-03-20',END_DATE)
        if w.empty: continue
        sub=d[d.venue.astype(str)==v][['game_id','datetime']].copy()
        sub['datetime']=pd.to_datetime(sub.datetime).dt.floor('h')
        w['datetime']=pd.to_datetime(w.datetime).dt.floor('h')
        sub=sub.merge(w,on='datetime',how='left')
        d=d.merge(sub[['game_id','weather_temp_c','weather_humidity_pct','weather_precip_mm','weather_wind_kmh']],on='game_id',how='left')
    # Save compact game rows using the *_pbp.csv naming expected by the engine.
    d['league']='NPB'; d['date']=pd.to_datetime(d.datetime); d['inning']=1; d['half']=''; d['event']=''; d['addedRuns']=0; d['pitcher']=''
    d.to_csv(OUT,index=False)
    official_audit()
    starters=((d.home_starter.fillna('').astype(str)!='')&(d.away_starter.fillna('').astype(str)!='')).sum()
    metric=((d.home_starter_era.notna())&(d.away_starter_era.notna())).sum()
    print(f'[OUTPUT] {OUT} games={len(d)} starters={starters} both_starter_lines={metric}')
    if len(d)<600 or starters<500: raise RuntimeError('insufficient NPB starter coverage')

if __name__=='__main__': main()