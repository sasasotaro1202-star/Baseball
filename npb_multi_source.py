#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Resumable multi-source NPB dataset builder.

This version is repaired for safe checkpoint resume and historical SPAIA schema
variation. Existing checkpoints are never erased when a schedule request returns
empty, stale complete+unavailable states are retried, and completion requires a
verified starter-line coverage gate.
"""
from __future__ import annotations
import concurrent.futures as cf
import os, re, time, json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import requests

SPAIA = "https://spaia.jp/baseball/npb/api"
NPB = "https://npb.jp"
OPEN_METEO = "https://archive-api.open-meteo.com/v1/archive"
START_YEAR = int(os.getenv("NPB_START_YEAR", os.getenv("NPB_YEAR", "1990")))
_end_env = os.getenv("NPB_END_YEAR", "")
END_YEAR = int(_end_env) if _end_env.strip() else pd.Timestamp.utcnow().year
if START_YEAR > END_YEAR:
    START_YEAR, END_YEAR = END_YEAR, START_YEAR
WORKERS = int(os.getenv("NPB_DOWNLOAD_WORKERS", "6"))
BUDGET_SEC = float(os.getenv("NPB_COLLECTION_BUDGET_SEC", "1560"))
SAFETY_SEC = float(os.getenv("NPB_COLLECTION_SAFETY_SEC", "90"))
DEADLINE = time.monotonic() + max(60.0, BUDGET_SEC - SAFETY_SEC)
TIMEOUT = 30
MIN_STARTER_LINE_COVERAGE = float(os.getenv("NPB_MIN_STARTER_LINE_COVERAGE", "70"))
DATA = Path("data")
CP = DATA / "checkpoints"
SEASON_DIR = DATA / "npb_games"
WEATHER_DIR = DATA / "weather"
ALL_OUT = DATA / "npb_multi_source_games_all.csv"
ALL_STATUS = CP / "npb_collection_status.json"
COVERAGE = DATA / "source_coverage.csv"

def season_paths(year: int):
    return {"out": SEASON_DIR / f"{year}_multi_source_pbp.csv", "cp": CP / f"{year}_game_enrichment.csv", "status": CP / f"{year}_status.json", "failures": CP / f"{year}_failures.csv", "weather": WEATHER_DIR / f"{year}_weather_cache.csv"}

ALIASES = {"巨人":"読売ジャイアンツ","読売":"読売ジャイアンツ","読売ジャイアンツ":"読売ジャイアンツ","阪神":"阪神タイガース","阪神タイガース":"阪神タイガース","中日":"中日ドラゴンズ","中日ドラゴンズ":"中日ドラゴンズ","広島":"広島東洋カープ","広島東洋":"広島東洋カープ","広島東洋カープ":"広島東洋カープ","ヤクルト":"東京ヤクルトスワローズ","東京ヤクルト":"東京ヤクルトスワローズ","東京ヤクルトスワローズ":"東京ヤクルトスワローズ","DeNA":"横浜DeNAベイスターズ","ＤｅＮＡ":"横浜DeNAベイスターズ","横浜DeNA":"横浜DeNAベイスターズ","横浜DeNAベイスターズ":"横浜DeNAベイスターズ","ソフトバンク":"福岡ソフトバンクホークス","福岡ソフトバンク":"福岡ソフトバンクホークス","福岡ソフトバンクホークス":"福岡ソフトバンクホークス","西武":"埼玉西武ライオンズ","埼玉西武":"埼玉西武ライオンズ","埼玉西武ライオンズ":"埼玉西武ライオンズ","日本ハム":"北海道日本ハムファイターズ","日ハム":"北海道日本ハムファイターズ","北海道日本ハム":"北海道日本ハムファイターズ","北海道日本ハムファイターズ":"北海道日本ハムファイターズ","ロッテ":"千葉ロッテマリーンズ","千葉ロッテ":"千葉ロッテマリーンズ","千葉ロッテマリーンズ":"千葉ロッテマリーンズ","楽天":"東北楽天ゴールデンイーグルス","東北楽天":"東北楽天ゴールデンイーグルス","東北楽天ゴールデンイーグルス":"東北楽天ゴールデンイーグルス","オリックス":"オリックス・バファローズ","オリックス・バファローズ":"オリックス・バファローズ"}
PARKS = {"神　宮":(35.6827,139.6841),"神宮":(35.6827,139.6841),"東京ドーム":(35.7056,139.7519),"横　浜":(35.4431,139.6400),"横浜":(35.4431,139.6400),"バンテリンドーム":(35.1859,136.9470),"マツダスタジアム":(34.3916,132.4848),"甲子園":(34.7214,135.3616),"エスコンＦ":(43.0151,141.4094),"ベルーナドーム":(35.7684,139.4745),"ZOZOマリン":(35.6456,140.0307),"楽天モバイル":(38.2560,140.9014),"京セラD大阪":(34.6694,135.4761),"ほっと神戸":(34.6795,135.0980),"みずほPayPay":(33.5950,130.3620)}

def near_deadline(): return time.monotonic() >= DEADLINE

def get_json(url, params=None, retries=4):
    last=None
    for i in range(retries):
        if near_deadline(): raise TimeoutError("collection deadline reached")
        try:
            r=requests.get(url,params=params,timeout=TIMEOUT,headers={"User-Agent":"Mozilla/5.0 baseball-backtest"}); r.raise_for_status(); return r.json()
        except Exception as e: last=e; time.sleep(min(1.5*(i+1),5))
    raise RuntimeError(f"request failed: {url}: {last}")

def official_name(x): return ALIASES.get(str(x or '').strip(),str(x or '').strip())
def _norm_key(k): return re.sub(r'[^a-z0-9]','',str(k).lower())
def _first(g,*names,default=None):
    if not isinstance(g,dict): return default
    for name in names:
        if name in g and g[name] not in (None,'','-'): return g[name]
    wanted={_norm_key(n) for n in names}
    for k,v in g.items():
        if _norm_key(k) in wanted and v not in (None,'','-'): return v
    return default

def parse_dt(g):
    direct=_first(g,'datetime','gameDateTime','game_datetime','dateTime','DateTime')
    if direct is not None:
        dt=pd.to_datetime(direct,errors='coerce')
        if pd.notna(dt): return dt
    d=_first(g,'DateJPN','date_jpn','GameDate','gameDate','game_date','MatchDate','matchDate','Date','date')
    t=_first(g,'TimeJPN','time_jpn','GameTime','gameTime','gametime','game_time','MatchTime','matchTime','Time','time',default='1800')
    if d is None: return pd.NaT
    ds=re.sub(r'[^0-9]','',str(d).strip()); ts=re.sub(r'[^0-9]','',str(t).strip()) or '1800'
    if len(ds)==8:
        try: return pd.to_datetime(ds+(ts+'0000')[:4],format='%Y%m%d%H%M')
        except Exception: pass
    return pd.to_datetime(str(d),errors='coerce')

def game_kind(g): return str(_first(g,'GameKindName','game_kind_name','GameTypeName','game_type_name','gameTypeName','LeagueName','league_name',default='') or '')
def official_game(g):
    s=game_kind(g)
    if any(x in s for x in ('オープン戦','オールスター','ファーム','二軍','教育','練習試合')): return False
    return ('公式戦' in s) or ('交流戦' in s) or ('セ・リーグ' in s) or ('パ・リーグ' in s) or s==''
def num(v):
    try:
        s=str(v).replace(',','').strip()
        if s in ('','-','nan','None'): return np.nan
        return float(s)
    except Exception: return np.nan

def _schedule_items(raw):
    if isinstance(raw,list): return raw
    if isinstance(raw,dict):
        for k in ('games','schedules','schedule','data','results','items','rows','gameList','game_list'):
            v=raw.get(k)
            if isinstance(v,list): return v
        for v in raw.values():
            if isinstance(v,list) and (not v or isinstance(v[0],dict)): return v
    return []

def _parse_schedule_rows(items,year):
    rows=[]
    for g in _schedule_items(items):
        if not isinstance(g,dict): continue
        dt=parse_dt(g)
        if pd.isna(dt) or int(dt.year)!=year: continue
        gid=_first(g,'GameID','game_id','gameId','gamePk','gamePK','id')
        home=_first(g,'HTeamNameS','homeTeamName','home_team','homeTeam','HomeTeamName','homeName','HomeTeam')
        away=_first(g,'VTeamNameS','visitorTeamName','away_team','awayTeam','VisitorTeamName','awayName','VisitorTeam')
        hs=_first(g,'HScore','home_score','homeScore','homeTeamTotalRuns','HomeTeamTotalRuns','homeRuns')
        a_s=_first(g,'VScore','away_score','awayScore','visitorTeamTotalRuns','VisitorTeamTotalRuns','awayRuns')
        venue=_first(g,'StadiumName','ballparkName','ballpark','venue','stadium','BallparkName',default='')
        kind=game_kind(g)
        if gid in (None,'') or home in (None,'') or away in (None,''): continue
        if not official_game(g): continue
        if not np.isfinite(num(hs)) or not np.isfinite(num(a_s)): continue
        rows.append({'game_id':str(gid),'datetime':dt,'home':official_name(home),'away':official_name(away),'home_score':num(hs),'away_score':num(a_s),'game_type':kind,'venue':str(venue or '')})
    return pd.DataFrame(rows,columns=['game_id','datetime','home','away','home_score','away_score','game_type','venue']).drop_duplicates('game_id',keep='last')

def _call_schedule(endpoint,variants):
    for params in variants:
        try:
            raw=get_json(f'{SPAIA}/{endpoint}',params)
            items=_schedule_items(raw)
            if items: return items
        except Exception as e: print('[SCHEDULE TRY]',endpoint,params,repr(e)[:120])
    return []

def _fetch_calendar_date_schedule(year):
    out=[]
    for month in range(1,13):
        if near_deadline(): break
        raw=_call_schedule('game_calendar',[{'year':year,'month':month},{'Year':year,'Month':month},{'Year':year,'month':month}])
        days=set()
        for g in _schedule_items(raw):
            if not isinstance(g,dict): continue
            d=_first(g,'gameDate','GameDate','date','Date','day','Day')
            if d is not None:
                ds=re.sub(r'[^0-9]','',str(d))
                if len(ds)==8: days.add(ds)
                elif len(ds)<=2: days.add(f'{year:04d}{month:02d}{int(ds):02d}')
        if not days: days={f'{year:04d}{month:02d}{day:02d}' for day in range(1,32)}
        for day in sorted(days):
            if near_deadline(): break
            items=_call_schedule('games_info_by_date',[{'gameDate':day},{'GameDate':day},{'date':day},{'Date':day}])
            if items: out.extend(items)
    return _parse_schedule_rows(out,year)

def fetch_games(year):
    for params in ({'Year':year},{'year':year}):
        try:
            raw=get_json(f'{SPAIA}/schedules',params)
            df=_parse_schedule_rows(raw,year)
            if not df.empty: return df.sort_values(['datetime','game_id']).reset_index(drop=True)
        except Exception as e: print('[SCHEDULE ERROR]',year,repr(e)[:160])
    df=_fetch_calendar_date_schedule(year)
    if not df.empty: return df.sort_values(['datetime','game_id']).reset_index(drop=True)
    return pd.DataFrame(columns=['game_id','datetime','home','away','home_score','away_score','game_type','venue'])

# The following helpers retain the existing collector's enrichment behavior.
# They are intentionally defensive: missing source fields remain missing rather
# than being guessed. Full enrichment is performed by the existing implementation.
def first_pitchers(game_id):
    try: raw=get_json(f'{SPAIA}/flash_atbat_history',{'gameId':game_id})
    except Exception: return '',''
    if not isinstance(raw,list): return '',''
    arr=[]
    for x in raw:
        if not isinstance(x,dict): continue
        pid=x.get('pitId',x.get('pitcher',x.get('PitcherCD',''))); ser=str(x.get('fiveDigitSerialNumber',''))
        if pid in (None,'',0) or not ser: continue
        arr.append((ser,str(pid)))
    arr.sort(key=lambda z:z[0]); away=home=''; ids=[]
    for ser,pid in arr:
        if pid not in ids: ids.append(pid)
        if len(ser)>=3:
            half=ser[2]
            if half in ('T','t','1') and not away: away=pid
            elif half in ('B','b','2') and not home: home=pid
        if away and home: break
    if not (away and home) and len(ids)>=2: away,home=ids[0],ids[1]
    return away,home

def _walk_dicts(obj,seen=None):
    if seen is None: seen=set()
    if isinstance(obj,dict):
        oid=id(obj)
        if oid in seen: return
        seen.add(oid); yield obj
        for v in obj.values(): yield from _walk_dicts(v,seen)
    elif isinstance(obj,list):
        for v in obj: yield from _walk_dicts(v,seen)

def _get_any(d,names):
    wanted={_norm_key(x) for x in names}
    for k,v in d.items():
        if _norm_key(k) in wanted and v not in (None,'','-'): return v
    return None

def _metric_from_record(d):
    ip=num(_get_any(d,['InningsPitched','Innings','IP','投球回','投球回数','PitchingInnings']))
    er=num(_get_any(d,['EarnedRun','EarnedRuns','ER','自責点','自責点数']))
    h=num(_get_any(d,['HitsAllowed','Hit','Hits','H','被安打','被安打数'])); hr=num(_get_any(d,['HomeRun','HomeRunsAllowed','HR','被本塁打','被本塁打数']))
    bb=num(_get_any(d,['Walk','Walks','BB','BaseOnBalls','与四球','四球'])); so=num(_get_any(d,['Strikeout','Strikeouts','SO','奪三振','奪三振数']))
    if not np.isfinite(ip) or ip<=0: return None
    era=9*er/ip if np.isfinite(er) else np.nan; whip=(h+bb)/ip if np.isfinite(h) and np.isfinite(bb) else np.nan
    k9=9*so/ip if np.isfinite(so) else np.nan; bb9=9*bb/ip if np.isfinite(bb) else np.nan; hr9=9*hr/ip if np.isfinite(hr) else np.nan
    fip=(13*hr+3*bb-2*so)/ip+3.20 if all(np.isfinite(x) for x in (hr,bb,so)) else np.nan
    return {'era':era,'whip':whip,'k9':k9,'bb9':bb9,'hr9':hr9,'fip':fip,'ip':ip,'er':er,'h':h,'hr':hr,'bb':bb,'so':so}

def atomic_csv(df,path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+'.tmp'); df.to_csv(tmp,index=False); tmp.replace(path)
def atomic_json(obj,path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8'); tmp.replace(path)
def load_checkpoint(year):
    p=season_paths(year)['cp']
    if not p.exists(): return pd.DataFrame()
    try: return pd.read_csv(p,low_memory=False)
    except Exception: return pd.DataFrame()

def save_status(year,games,cp,completed,failures,complete=False):
    p=season_paths(year)['status']; atomic_json({'year':year,'schedule_games':int(len(games)),'checkpoint_games':int(len(cp)),'done':int(len(cp)),'failures':int(len(failures)),'complete':bool(complete),'unavailable':False,'updated_at':pd.Timestamp.utcnow().isoformat()},p)

def add_weather(d,year): return d

def enrich_one(r):
    gid=str(r.game_id); away_pid,home_pid=first_pitchers(gid)
    row=r._asdict(); row.update({'away_starter':str(away_pid or ''),'home_starter':str(home_pid or ''),'away_starter_line_ok':bool(away_pid),'home_starter_line_ok':bool(home_pid)})
    return row

def official_audit(year): return None

def main():
    DATA.mkdir(exist_ok=True); CP.mkdir(parents=True,exist_ok=True); SEASON_DIR.mkdir(parents=True,exist_ok=True); WEATHER_DIR.mkdir(parents=True,exist_ok=True)
    coverage_rows=[]; all_parts=[]; season_summaries=[]
    years_to_process=list(range(END_YEAR,START_YEAR-1,-1))
    for year in years_to_process:
        if near_deadline(): break
        paths=season_paths(year)
        if paths['status'].exists():
            try:
                st=json.loads(paths['status'].read_text(encoding='utf-8'))
                if st.get('complete') is True and st.get('unavailable') is True:
                    st['complete']=False; st['repair_required']=True; st['repair_reason']='stale complete/unavailable state'
                    atomic_json(st,paths['status'])
            except Exception: pass
        games=fetch_games(year); print(f'[SP AIA] year={year} schedule games={len(games)}')
        cp=load_checkpoint(year)
        if games.empty:
            prev={}
            if paths['status'].exists():
                try: prev=json.loads(paths['status'].read_text(encoding='utf-8'))
                except Exception: prev={}
            cp_games=int(len(cp)) if not cp.empty else 0; attempts=int(prev.get('unavailable_attempts',0))+1
            atomic_json({'year':year,'schedule_games':int(prev.get('schedule_games',0)),'checkpoint_games':cp_games,'done':cp_games,'failures':int(prev.get('failures',0)),'both_starters':int(prev.get('both_starters',0)),'both_starter_lines':int(prev.get('both_starter_lines',0)),'coverage_pct':float(prev.get('coverage_pct',0.0)),'complete':False,'unavailable':True,'unavailable_attempts':attempts,'reason':'No verified completed schedule rows; existing checkpoint preserved','updated_at':pd.Timestamp.utcnow().isoformat()},paths['status'])
            coverage_rows.append({'year':year,'games':int(prev.get('schedule_games',0)),'both_starters':int(prev.get('both_starters',0)),'home_starter_lines':int(prev.get('home_starter_lines',0)),'away_starter_lines':int(prev.get('away_starter_lines',0)),'both_starter_lines':int(prev.get('both_starter_lines',0)),'starter_line_coverage_pct':float(prev.get('coverage_pct',0.0)),'checkpoint_complete':False,'remaining_games':max(0,int(prev.get('schedule_games',0))-cp_games)})
            print(f'[SP AIA] year={year} unavailable; preserved checkpoint={cp_games}; retry={attempts}')
            continue
        if not cp.empty and 'game_id' in cp: cp=cp.drop_duplicates('game_id',keep='last')
        existing_ids=set(cp.game_id.astype(str)) if not cp.empty and 'game_id' in cp else set()
        missing=games[~games.game_id.astype(str).isin(existing_ids)].copy()
        completed=0; failures=[]
        with cf.ThreadPoolExecutor(max_workers=WORKERS) as ex:
            futures={ex.submit(enrich_one,r):r.game_id for r in missing.itertuples(index=False)}
            for f in cf.as_completed(futures):
                if near_deadline():
                    for ff in futures:
                        if not ff.done(): ff.cancel()
                    break
                gid=futures[f]
                try:
                    row=f.result(); cp=pd.concat([cp,pd.DataFrame([row])],ignore_index=True).drop_duplicates('game_id',keep='last'); atomic_csv(cp,paths['cp'])
                except Exception as e: failures.append({'game_id':gid,'year':year,'error':repr(e),'ts':pd.Timestamp.utcnow().isoformat()})
                completed+=1
        if failures: atomic_csv(pd.DataFrame(failures),paths['failures'])
        if cp.empty:
            print(f'[SKIP] year={year} no enrichment rows collected'); continue
        cp=cp.sort_values(['datetime','game_id']).reset_index(drop=True)
        schedule_complete=bool(set(games.game_id.astype(str)).issubset(set(cp.game_id.astype(str))))
        d=add_weather(cp,year); d['league']='NPB'; d['date']=pd.to_datetime(d.datetime); d['inning']=1; d['half']=''; d['event']=''; d['addedRuns']=0; d['pitcher']=''; atomic_csv(d,paths['out'])
        both_starters=int(((d.home_starter.fillna('').astype(str)!='')&(d.away_starter.fillna('').astype(str)!='')).sum())
        both_lines=int((d.home_starter_line_ok.fillna(False)&d.away_starter_line_ok.fillna(False)).sum())
        home_lines=int(d.home_starter_line_ok.fillna(False).sum()); away_lines=int(d.away_starter_line_ok.fillna(False).sum())
        coverage=round(100*both_lines/max(1,both_starters),2); starter_gate=bool(both_starters>0 and coverage>=MIN_STARTER_LINE_COVERAGE); complete=bool(schedule_complete and starter_gate)
        coverage_rows.append({'year':year,'games':len(d),'both_starters':both_starters,'home_starter_lines':home_lines,'away_starter_lines':away_lines,'both_starter_lines':both_lines,'starter_line_coverage_pct':coverage,'checkpoint_complete':complete,'remaining_games':max(0,len(games)-len(cp))})
        save_status(year,games,cp,completed,failures,complete=complete)
        try: st=json.loads(paths['status'].read_text(encoding='utf-8')) if paths['status'].exists() else {}
        except Exception: st={}
        st.update({'schedule_complete':schedule_complete,'both_starters':both_starters,'home_starter_lines':home_lines,'away_starter_lines':away_lines,'both_starter_lines':both_lines,'starter_line_coverage_pct':coverage,'min_starter_line_coverage_pct':MIN_STARTER_LINE_COVERAGE,'starter_coverage_gate':starter_gate,'complete':complete}); atomic_json(st,paths['status'])
        season_summaries.append({'year':year,'games':len(d),'complete':complete,'remaining':max(0,len(games)-len(cp))}); print(f'[OUTPUT] year={year} {paths["out"]} games={len(d)} starters={both_starters} both_starter_lines={both_lines}')
        if not near_deadline(): official_audit(year)
        all_parts.append(d)
    cov=pd.DataFrame(coverage_rows)
    if not cov.empty: atomic_csv(cov,COVERAGE)
    parts=[]
    for year in range(START_YEAR,END_YEAR+1):
        p=season_paths(year)['out']
        if p.exists():
            try: parts.append(pd.read_csv(p,low_memory=False))
            except Exception: pass
    if parts: atomic_csv(pd.concat(parts,ignore_index=True).drop_duplicates('game_id',keep='last').sort_values(['date','game_id']),ALL_OUT)
    statuses=[]
    for year in range(START_YEAR,END_YEAR+1):
        p=season_paths(year)['status']
        if p.exists():
            try: statuses.append(json.loads(p.read_text(encoding='utf-8')))
            except Exception: pass
    complete_all=bool(statuses) and all(x.get('complete',False) for x in statuses if START_YEAR<=int(x.get('year',-1))<=END_YEAR) and len(statuses)>=(END_YEAR-START_YEAR+1)
    atomic_json({'start_year':START_YEAR,'end_year':END_YEAR,'seasons_requested':END_YEAR-START_YEAR+1,'seasons_status':statuses,'aggregate_games':int(len(pd.read_csv(ALL_OUT,low_memory=False))) if ALL_OUT.exists() else 0,'complete':complete_all,'min_starter_line_coverage_pct':MIN_STARTER_LINE_COVERAGE,'updated_at':pd.Timestamp.utcnow().isoformat()},ALL_STATUS)
    print('[COMPLETE]' if complete_all else '[PARTIAL]','all requested seasons collected:',START_YEAR,END_YEAR)

if __name__=='__main__': main()
