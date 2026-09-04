#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Idempotent runtime hardening for the NPB collector."""
from __future__ import annotations
import html
import os
import re
from pathlib import Path

P = Path("npb_multi_source.py")
s = P.read_text(encoding="utf-8")
if "# RUNTIME_HARDENING_V3" in s:
    print("[RUNTIME PATCH] V3 already applied")
    raise SystemExit(0)

anchor = 'SAFETY_SEC = float(os.getenv("NPB_COLLECTION_SAFETY_SEC", "90"))\n'
if 'MIN_STARTER_LINE_COVERAGE' not in s:
    if anchor not in s: raise RuntimeError("config anchor not found")
    s = s.replace(anchor, anchor + 'MIN_STARTER_LINE_COVERAGE = float(os.getenv("NPB_MIN_STARTER_LINE_COVERAGE", "70"))\n', 1)

pat = re.compile(r"def first_pitchers\(game_id\):.*?\n\ndef _norm_key", re.S)
new = '''def _official_starters_from_npb(game_id, home="", away=""):
    team_codes={
      "読売ジャイアンツ":"g","東京ヤクルトスワローズ":"s","横浜DeNAベイスターズ":"db",
      "広島東洋カープ":"c","阪神タイガース":"t","中日ドラゴンズ":"d",
      "福岡ソフトバンクホークス":"h","埼玉西武ライオンズ":"l","北海道日本ハムファイターズ":"f",
      "千葉ロッテマリーンズ":"m","東北楽天ゴールデンイーグルス":"e","オリックス・バファローズ":"bs",
    }
    gid=str(game_id).replace(".0","")
    if len(gid)<10 or not gid[:8].isdigit(): return "",""
    y,md,no=gid[:4],gid[4:8],gid[-2:]
    ac=team_codes.get(official_name(away)); hc=team_codes.get(official_name(home))
    if not ac or not hc: return "",""
    url=f"{NPB}/scores/{y}/{md}/{ac}-{hc}-{no}/playbyplay.html"
    try:
        r=requests.get(url,timeout=12,headers={"User-Agent":"Mozilla/5.0 baseball-backtest"})
        if r.status_code!=200: return "",""
        text=re.sub(r"<script.*?</script>|<style.*?</style>"," ",r.text,flags=re.I|re.S)
        text=html.unescape(re.sub(r"<[^>]+>"," ",text))
        text=re.sub(r"\s+"," ",text)
        names=re.findall(r"（先発投手）\s*([^\s|]+)",text)
        if len(names)>=2: return names[0].strip(),names[1].strip()
    except Exception:
        pass
    return "",""

def first_pitchers(game_id, home="", away=""):
    try:
        raw=get_json(f'{SPAIA}/flash_atbat_history',{'gameId':game_id})
    except Exception:
        raw=None
    if isinstance(raw,list):
        arr=[]
        for x in raw:
            if not isinstance(x,dict): continue
            pid=x.get('pitId',x.get('pitcher',x.get('PitcherCD','')))
            ser=str(x.get('fiveDigitSerialNumber',''))
            if pid in (None,'',0) or not ser: continue
            arr.append((ser,str(pid)))
        arr.sort(key=lambda z:z[0]); away_id=home_id=''; seen=set()
        for ser,pid in arr:
            if len(ser)>=3:
                half=ser[2]; inning=ser[:2]; key=(inning,half)
                if key in seen: continue
                seen.add(key)
                if half in ('T','t','1') and not away_id: away_id=pid
                elif half in ('B','b','2') and not home_id: home_id=pid
            if away_id and home_id: break
        if away_id and home_id: return away_id,home_id
    return _official_starters_from_npb(game_id,home,away)

def _norm_key'''
s,n=pat.subn(lambda m:new,s,count=1)
if n!=1: raise RuntimeError("first_pitchers target not found")
s = s.replace('away,home=first_pitchers(r.game_id); date=', 'away,home=first_pitchers(r.game_id,r.home,r.away); date=', 1)

pat = re.compile(r"def pitcher_line\(game_id,match_date,starter_id\):.*?\n\n# ---------------------------------------------------------------------------", re.S)
new = '''def pitcher_line(game_id,match_date,starter_id):
    recs,err=_fetch_game_pitcher_records(game_id,match_date); sid=str(starter_id).strip()
    if not sid: return None
    for d in recs:
        pid=_record_pid(d)
        if pid is not None and str(pid)==sid:
            m=_metric_from_record(d)
            if m: return m
    for d in recs:
        name=_get_any(d,NAME_KEYS) if 'NAME_KEYS' in globals() else None
        if name and str(name).strip()==sid:
            m=_metric_from_record(d)
            if m: return m
    try:
        sn=int(float(sid))
        for d in recs:
            pid=_record_pid(d)
            try:
                if int(float(str(pid)))==sn:
                    m=_metric_from_record(d)
                    if m: return m
            except Exception: pass
    except Exception: pass
    return None

# ---------------------------------------------------------------------------'''
s,n=pat.subn(lambda m:new,s,count=1)
if n!=1: raise RuntimeError("pitcher_line target not found")

pat = re.compile(r"        if games\.empty:\n.*?        if not cp\.empty and 'game_id' in cp:", re.S)
new = '''        if games.empty:
            if not cp.empty and 'game_id' in cp:
                prev_games=len(cp)
                both_starters=int(((cp.get('home_starter','').fillna('').astype(str)!='')&(cp.get('away_starter','').fillna('').astype(str)!='')).sum()) if 'home_starter' in cp and 'away_starter' in cp else 0
                both_lines=int((cp.get('home_starter_line_ok',pd.Series(dtype=bool)).fillna(False)&cp.get('away_starter_line_ok',pd.Series(dtype=bool)).fillna(False)).sum()) if 'home_starter_line_ok' in cp and 'away_starter_line_ok' in cp else 0
                coverage=round(100*both_lines/max(1,both_starters),2)
                atomic_json({'year':year,'schedule_games':prev_games,'checkpoint_games':prev_games,'done':0,'failures':0,'both_starters':both_starters,'both_starter_lines':both_lines,'coverage_pct':coverage,'complete':False,'unavailable':False,'schedule_complete':False,'updated_at':pd.Timestamp.utcnow().isoformat()},paths['status'])
                print(f'[SP AIA] year={year} EMPTY SCHEDULE -> preserved checkpoint rows={prev_games}; NOT complete')
                continue
            atomic_json({'year':year,'schedule_games':0,'checkpoint_games':0,'done':0,'failures':0,'both_starters':0,'both_starter_lines':0,'coverage_pct':0.0,'complete':False,'unavailable':True,'schedule_complete':False,'updated_at':pd.Timestamp.utcnow().isoformat()},paths['status'])
            print(f'[SP AIA] year={year} no schedule rows; marked unavailable/INCOMPLETE')
            continue
        if not cp.empty and 'game_id' in cp:'''
s,n=pat.subn(lambda m:new,s,count=1)
if n!=1: raise RuntimeError("empty schedule target not found")

s=s.replace('save_status(year,games,cp,completed,failures,complete=complete)', 'save_status(year,games,cp,completed,failures,complete=(complete and coverage >= MIN_STARTER_LINE_COVERAGE))',1)
s=s.replace('bad=eligible[eligible.starter_line_coverage_pct < 70]', 'bad=eligible[eligible.starter_line_coverage_pct < MIN_STARTER_LINE_COVERAGE]',1)
s=s.replace("complete_all=bool(statuses) and all(x.get('complete',False) for x in statuses if START_YEAR <= int(x.get('year',-1)) <= END_YEAR) and len(statuses) >= (END_YEAR-START_YEAR+1)", "complete_all=(len(statuses) >= (END_YEAR-START_YEAR+1) and all((START_YEAR <= int(x.get('year',-1)) <= END_YEAR) and bool(x.get('complete',False)) and not bool(x.get('unavailable',False)) and float(x.get('coverage_pct',0.0)) >= MIN_STARTER_LINE_COVERAGE for x in statuses))",1)
s='# RUNTIME_HARDENING_V3\n'+s
P.write_text(s,encoding='utf-8')
print('[RUNTIME PATCH] V3 applied: official NPB starter fallback + no guessing + non-destructive checkpoint + coverage gate')
