#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime hardening for the NPB collector.

This deliberately patches the checked-in rich collector in-place immediately
before execution. It keeps the large historical collector intact while making
critical safety fixes without rewriting its full source through the API.
"""
from __future__ import annotations
import os
import re
from pathlib import Path

P = Path("npb_multi_source.py")
s = P.read_text(encoding="utf-8")

# 1) Never infer a starting pitcher from arbitrary first pitcher IDs.
pat = re.compile(r"def first_pitchers\(game_id\):.*?\n\ndef _norm_key", re.S)
new = '''def first_pitchers(game_id):
    raw=get_json(f'{SPAIA}/flash_atbat_history',{'gameId':game_id})
    if not isinstance(raw,list): return '',''
    arr=[]
    for x in raw:
        if not isinstance(x,dict): continue
        pid=x.get('pitId',x.get('pitcher',x.get('PitcherCD','')))
        ser=str(x.get('fiveDigitSerialNumber',''))
        if pid in (None,'',0) or not ser: continue
        arr.append((ser,str(pid)))
    arr.sort(key=lambda z:z[0]); away=home=''; seen=set()
    for ser,pid in arr:
        if len(ser)>=3:
            half=ser[2]; inning=ser[:2]; key=(inning,half)
            if key in seen: continue
            seen.add(key)
            if half in ('T','t','1') and not away: away=pid
            elif half in ('B','b','2') and not home: home=pid
        if away and home: break
    # If the API cannot identify both sides unambiguously, return missing.
    # Never guess from arbitrary pitcher order.
    return away,home

def _norm_key'''
s,n=pat.subn(new,s,count=1)
if n!=1: raise RuntimeError('first_pitchers patch target not found')

# 2) Configurable starter-line quality gate.
if 'MIN_STARTER_LINE_COVERAGE' not in s:
    anchor='SAFETY_SEC = float(os.getenv("NPB_COLLECTION_SAFETY_SEC", "90"))\n'
    if anchor not in s: raise RuntimeError('config anchor not found')
    s=s.replace(anchor,anchor+'MIN_STARTER_LINE_COVERAGE = float(os.getenv("NPB_MIN_STARTER_LINE_COVERAGE", "70"))\n',1)

# 3) Empty schedule must never erase a non-empty checkpoint. A genuine upstream
#    unavailable season remains incomplete rather than being falsely complete.
pat = re.compile(r"        if games\.empty:\n.*?        if not cp\.empty and 'game_id' in cp:", re.S)
new = '''        if games.empty:
            if not cp.empty and 'game_id' in cp:
                # Preserve prior checkpoint data when the upstream schedule API
                # temporarily returns an empty/unparseable response.
                prev_games=len(cp)
                both_starters=int(((cp.get('home_starter','').fillna('').astype(str)!='')&(cp.get('away_starter','').fillna('').astype(str)!='')).sum()) if 'home_starter' in cp and 'away_starter' in cp else 0
                both_lines=int((cp.get('home_starter_line_ok',pd.Series(dtype=bool)).fillna(False)&cp.get('away_starter_line_ok',pd.Series(dtype=bool)).fillna(False)).sum()) if 'home_starter_line_ok' in cp and 'away_starter_line_ok' in cp else 0
                coverage=round(100*both_lines/max(1,both_starters),2)
                atomic_json({'year':year,'schedule_games':prev_games,'checkpoint_games':prev_games,'done':0,'failures':0,
                              'both_starters':both_starters,'both_starter_lines':both_lines,'coverage_pct':coverage,
                              'complete':False,'unavailable':False,'schedule_complete':False,
                              'updated_at':pd.Timestamp.utcnow().isoformat()},paths['status'])
                coverage_rows.append({'year':year,'games':prev_games,'both_starters':both_starters,
                                      'home_starter_lines':int(cp.get('home_starter_line_ok',pd.Series(dtype=bool)).fillna(False).sum()),
                                      'away_starter_lines':int(cp.get('away_starter_line_ok',pd.Series(dtype=bool)).fillna(False).sum()),
                                      'both_starter_lines':both_lines,'starter_line_coverage_pct':coverage,
                                      'checkpoint_complete':False,'remaining_games':0})
                print(f'[SP AIA] year={year} EMPTY SCHEDULE -> preserved checkpoint rows={prev_games}; NOT complete')
                continue
            # No prior data exists. Do not fabricate a completed season.
            atomic_json({'year':year,'schedule_games':0,'checkpoint_games':0,'done':0,'failures':0,
                          'both_starters':0,'both_starter_lines':0,'coverage_pct':0.0,
                          'complete':False,'unavailable':True,'schedule_complete':False,
                          'updated_at':pd.Timestamp.utcnow().isoformat()},paths['status'])
            coverage_rows.append({'year':year,'games':0,'both_starters':0,'home_starter_lines':0,'away_starter_lines':0,
                                  'both_starter_lines':0,'starter_line_coverage_pct':0.0,
                                  'checkpoint_complete':False,'remaining_games':0})
            print(f'[SP AIA] year={year} no schedule rows; marked unavailable/INCOMPLETE')
            continue
        if not cp.empty and 'game_id' in cp:'''
s,n=pat.subn(new,s,count=1)
if n!=1: raise RuntimeError('empty-schedule patch target not found')

# 4) A season is complete only when schedule is complete AND both-starter line
#    coverage reaches the configured threshold.
old='save_status(year,games,cp,completed,failures,complete=complete)'
new='save_status(year,games,cp,completed,failures,complete=(complete and coverage >= MIN_STARTER_LINE_COVERAGE))'
if old not in s: raise RuntimeError('save_status call target not found')
s=s.replace(old,new,1)

# 5) Persist explicit schedule_complete in the season status.
old="atomic_json({'year':year,'schedule_games':len(games),'checkpoint_games':len(cp),'done':int(done),'failures':len(failures),'both_starters':both_starters,'both_starter_lines':both_lines,'coverage_pct':round(100*both_lines/max(1,both_starters),2),'complete':bool(complete),'updated_at':pd.Timestamp.utcnow().isoformat()},paths[\"status\"])"
new="atomic_json({'year':year,'schedule_games':len(games),'checkpoint_games':len(cp),'done':int(done),'failures':len(failures),'both_starters':both_starters,'both_starter_lines':both_lines,'coverage_pct':round(100*both_lines/max(1,both_starters),2),'schedule_complete':bool(complete),'complete':bool(complete and (100*both_lines/max(1,both_starters)) >= MIN_STARTER_LINE_COVERAGE),'updated_at':pd.Timestamp.utcnow().isoformat()},paths[\"status\"])"
if old not in s: raise RuntimeError('save_status definition target not found')
s=s.replace(old,new,1)

# 6) Global completion must exclude unavailable/partial seasons.
old="complete_all=bool(statuses) and all(x.get('complete',False) for x in statuses if START_YEAR <= int(x.get('year',-1)) <= END_YEAR) and len(statuses) >= (END_YEAR-START_YEAR+1)"
new="complete_all=(len(statuses) >= (END_YEAR-START_YEAR+1) and all((START_YEAR <= int(x.get('year',-1)) <= END_YEAR) and bool(x.get('complete',False)) and not bool(x.get('unavailable',False)) and bool(x.get('schedule_complete',x.get('complete',False))) and float(x.get('coverage_pct',0.0)) >= MIN_STARTER_LINE_COVERAGE for x in statuses))"
if old not in s: raise RuntimeError('complete_all target not found')
s=s.replace(old,new,1)

# 7) Final hard gate uses the configured threshold, not a duplicated literal.
s=s.replace("bad=eligible[eligible.starter_line_coverage_pct < 70]","bad=eligible[eligible.starter_line_coverage_pct < MIN_STARTER_LINE_COVERAGE]",1)

# 8) Stamp the active hardening version for auditability.
marker='# RUNTIME_HARDENING_V2'
if marker not in s:
    s=marker+'\n'+s

P.write_text(s,encoding='utf-8')
print('[RUNTIME PATCH] npb_multi_source.py hardened: no starter guessing, non-destructive empty schedule, schedule/complete split, 70% gate')
