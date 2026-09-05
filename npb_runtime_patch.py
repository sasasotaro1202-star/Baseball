#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Idempotent NPB collector hardening and source-hierarchy patch.

Source hierarchy used by the collector:
1. NPB.jp official pages for canonical schedule/result validation and official
   starting-pitcher fallback.
2. SPAIA game APIs for granular PBP, player-game, pitcher-game and lineup data.
3. Open-Meteo archive for historical weather.

No source is silently substituted with guessed values. Missing fields remain
missing and keep the season below the completion gate.
"""
from __future__ import annotations
import html
import os
import re
from pathlib import Path

P = Path("npb_multi_source.py")
s = P.read_text(encoding="utf-8")
if "# RUNTIME_HARDENING_V5" in s:
    print("[RUNTIME PATCH] V5 already applied")
    raise SystemExit(0)

# Keep the existing V4 hardening if the base collector has not been patched yet.
if "# RUNTIME_HARDENING_V4" not in s:
    anchor = 'SAFETY_SEC = float(os.getenv("NPB_COLLECTION_SAFETY_SEC", "90"))\n'
    if 'MIN_STARTER_LINE_COVERAGE' not in s:
        if anchor not in s:
            raise RuntimeError("collector config anchor not found")
        s = s.replace(anchor, anchor + 'MIN_STARTER_LINE_COVERAGE = float(os.getenv("NPB_MIN_STARTER_LINE_COVERAGE", "70"))\n', 1)

    pat = re.compile(r"def first_pitchers\(game_id\):.*?\n\ndef _norm_key", re.S)
    new = '''def _official_starters_from_npb(game_id, home="", away=""):
    team_codes={
      "読売ジャイアンツ":"g","東京ヤクルトスワローズ":"s","横浜DeNAベイスターズ":"db",
      "広島東洋カープ":"c","阪神タイガース":"t","中日ドラゴンズ":"d",
      "福岡ソフトバンクホークス":"h","埼玉西武ライオンズ":"l","北海道日本ハムファイターズ":"f",
      "千葉ロッテマリーンズ":"m","東北楽天ゴールデンイーグルス":"e","オリックス・バファローズ":"bs",
    }
    gid=re.sub(r"\\.0$", "", str(game_id).strip())
    if len(gid)<10 or not gid[:8].isdigit(): return "",""
    y,md,no=gid[:4],gid[4:8],gid[-2:]
    ac=team_codes.get(official_name(away)); hc=team_codes.get(official_name(home))
    if not ac or not hc: return "",""
    url=f"{NPB}/scores/{y}/{md}/{ac}-{hc}-{no}/playbyplay.html"
    try:
        r=requests.get(url,timeout=12,headers={"User-Agent":"Mozilla/5.0 baseball-backtest"})
        if r.status_code!=200: return "",""
        text=html.unescape(re.sub(r"<[^>]+>"," ",re.sub(r"<script.*?</script>|<style.*?</style>"," ",r.text,flags=re.I|re.S)))
        text=re.sub(r"\\s+"," ",text)
        names=re.findall(r"(?:\\(先発投手\\)|（先発投手）)\\s*([^<|\\s]+)",text)
        if len(names)>=2: return names[0].strip(),names[1].strip()
        names=re.findall(r"先発投手\\)?）?\\s*[:：]?\\s*([^<|\\s]+)",text)
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
    # Strict rule: unresolved starters remain unresolved; never pick arbitrary pitchers.
    return _official_starters_from_npb(game_id,home,away)

def _norm_key'''
    s,n=pat.subn(lambda m:new,s,count=1)
    if n != 1:
        raise RuntimeError("first_pitchers replacement target not found")
    s=s.replace('away,home=first_pitchers(r.game_id); date=', 'away,home=first_pitchers(r.game_id,r.home,r.away); date=', 1)

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
    if n != 1:
        raise RuntimeError("empty schedule replacement target not found")
    s=s.replace('save_status(year,games,cp,completed,failures,complete=complete)', 'save_status(year,games,cp,completed,failures,complete=(complete and coverage >= MIN_STARTER_LINE_COVERAGE))', 1)
    s=s.replace('bad=eligible[eligible.starter_line_coverage_pct < 70]', 'bad=eligible[eligible.starter_line_coverage_pct < MIN_STARTER_LINE_COVERAGE]', 1)
    s=s.replace("complete_all=bool(statuses) and all(x.get('complete',False) for x in statuses if START_YEAR <= int(x.get('year',-1)) <= END_YEAR) and len(statuses) >= (END_YEAR-START_YEAR+1)", "complete_all=(len(statuses) >= (END_YEAR-START_YEAR+1) and all((START_YEAR <= int(x.get('year',-1)) <= END_YEAR) and bool(x.get('complete',False)) and not bool(x.get('unavailable',False)) and float(x.get('coverage_pct',0.0)) >= MIN_STARTER_LINE_COVERAGE for x in statuses))", 1)

# V5: checkpoint quality is now based on prediction-critical enrichment, not
# merely on whether a row was written previously. This specifically repairs
# stale checkpoints such as the old 2017 12-row/0-starter state.
marker = "# RUNTIME_HARDENING_V5\n"
needle = "        missing_ids=set(games.game_id.astype(str)) - player_done_ids\n        missing=games[games.game_id.astype(str).isin(missing_ids)].copy()\n        print(f'[CHECKPOINT] year={year} existing={len(existing_ids)} player_enriched={len(player_done_ids)} missing_for_full_enrichment={len(missing)}')"
replacement = '''        # Re-enrich any checkpoint row whose prediction-critical data is incomplete.
        # A previously written row is NOT considered done when either starter or
        # either starter game line is missing. Player-level enrichment is also
        # required, but never allowed to suppress starter repair.
        needs_ids=set()
        for gid in games.game_id.astype(str):
            if gid not in existing_ids:
                needs_ids.add(gid); continue
            rr=cp[cp.game_id.astype(str)==gid].iloc[-1]
            hs=str(rr.get('home_starter','') or '').strip(); aw=str(rr.get('away_starter','') or '').strip()
            hl=bool(rr.get('home_starter_line_ok',False)); al=bool(rr.get('away_starter_line_ok',False))
            if not (hs and aw and hl and al):
                needs_ids.add(gid); continue
            if gid not in player_done_ids:
                needs_ids.add(gid)
        missing=games[games.game_id.astype(str).isin(needs_ids)].copy()
        print(f'[CHECKPOINT] year={year} existing={len(existing_ids)} player_enriched={len(player_done_ids)} repair_or_missing={len(missing)}')'''
if needle not in s:
    raise RuntimeError("checkpoint enrichment target not found")
s=s.replace(needle,replacement,1)

# Add source provenance fields to every enriched game without changing the
# feature values. This makes the dataset auditable by source and failure mode.
needle2 = "    z['home_starter_line_ok']=bool(hm); z['away_starter_line_ok']=bool(am)\n    z['_player_rows']=players"
replacement2 = "    z['home_starter_line_ok']=bool(hm); z['away_starter_line_ok']=bool(am)\n    z['starter_source']='SPAIA game API + NPB.jp official fallback'\n    z['player_source']='SPAIA game API'\n    z['weather_source']='Open-Meteo archive'\n    z['_player_rows']=players"
if needle2 not in s:
    raise RuntimeError("provenance target not found")
s=s.replace(needle2,replacement2,1)

s = marker + s
P.write_text(s,encoding='utf-8')
print('[RUNTIME PATCH] V5 applied: source hierarchy + stale-checkpoint repair + strict official starter fallback + coverage gate')
