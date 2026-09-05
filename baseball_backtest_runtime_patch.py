#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Idempotent runtime hardening for NPB collector/backtest integration."""
from __future__ import annotations
import re
from pathlib import Path
P=Path('baseball_backtest.py')
s=P.read_text(encoding='utf-8')
if '# BACKTEST_RUNTIME_HARDENING_V2' in s:
    print('[BACKTEST PATCH] V2 already applied'); raise SystemExit(0)

anchor='    def aggregate_npb_games(self, pbp: pd.DataFrame) -> pd.DataFrame:\n'
method='''    def _normalize_npb_pbp(self, df: pd.DataFrame) -> pd.DataFrame:\n        d=df.copy()\n        aliases={"gameId":"game_id","GameID":"game_id","gameDate":"datetime","DateJPN":"datetime","GameDate":"datetime","homeTeam":"home","awayTeam":"away","homeScore":"home_score","awayScore":"away_score","GameKindName":"game_type","GameTypeName":"game_type","ballparkName":"venue"}\n        rename={c:aliases[c] for c in d.columns if c in aliases and aliases[c] not in d.columns}\n        if rename: d=d.rename(columns=rename)\n        if 'game_id' not in d.columns: raise RuntimeError('NPB data has no game_id')\n        d['game_id']=d['game_id'].astype(str).str.replace(r'\\.0$','',regex=True)\n        if 'datetime' not in d.columns: d['datetime']=d.get('date',pd.NaT)\n        d['datetime']=pd.to_datetime(d['datetime'],errors='coerce',utc=True)\n        if 'date' not in d.columns: d['date']=d['datetime']\n        else: d['date']=pd.to_datetime(d['date'],errors='coerce',utc=True).fillna(d['datetime'])\n        for c in ('home','away','game_type','venue'):\n            if c not in d.columns: d[c]=''\n        for c in ('home_score','away_score'):\n            if c not in d.columns: d[c]=np.nan\n            d[c]=pd.to_numeric(d[c],errors='coerce')\n        if 'row_order' not in d.columns: d['row_order']=d.groupby('game_id',sort=False).cumcount()\n        return d.sort_values(['datetime','game_id','row_order'],na_position='last').reset_index(drop=True)\n\n'''
if anchor not in s: raise RuntimeError('aggregate anchor missing')
s=s.replace(anchor,method+anchor,1)
old='''            hp = self._first_pitcher(g, "home")\n            ap = self._first_pitcher(g, "away")'''
new='''            hp = self._first_value(g, "home_starter") or self._first_pitcher(g, "home")\n            ap = self._first_value(g, "away_starter") or self._first_pitcher(g, "away")'''
if old in s:
    s=s.replace(old,new,1)
    anchor='    def _first_pitcher(self, g: pd.DataFrame, side: str) -> str:\n'
    helper='''    def _first_value(self, g: pd.DataFrame, col: str) -> str:\n        if col not in g: return ""\n        for x in g[col].tolist():\n            v=str(x).strip()\n            if v and v.lower() not in ("nan","none"): return v\n        return ""\n\n'''
    if anchor not in s: raise RuntimeError('first pitcher anchor missing')
    s=s.replace(anchor,helper+anchor,1)
needle='''                "home_starter": hp, "away_starter": ap,\n                "venue": "unknown", "confirmed_starters": bool(hp and ap),\n            })'''
repl='''                "home_starter": hp, "away_starter": ap,\n                "venue": str(g["venue"].dropna().iloc[0]) if "venue" in g and g["venue"].notna().any() else "unknown",\n                "confirmed_starters": bool(hp and ap),\n                **{c: g[c].dropna().iloc[0] for c in g.columns if c.startswith(("home_starter_","away_starter_")) and c not in ("home_starter_line_ok","away_starter_line_ok") and g[c].notna().any()},\n            })'''
if needle in s: s=s.replace(needle,repl,1)
pat=re.compile(r'    def _update_pitcher_history\(self, row: pd.Series\):.*?\n\n    # ------------------------------------------------------------------\n    # Models',re.S)
new='''    def _update_pitcher_history(self, row: pd.Series):\n        for side in ("home", "away"):\n            p=str(row.get(f"{side}_starter","") or "").strip()\n            if not p or p.lower() in ("nan","none"): continue\n            cols={}\n            for m in ("era","whip","k9","bb9","hr9","fip","pitches","k_rate","bb_rate","ip","er","h","hr","bb","so"):\n                v=row.get(f"{side}_starter_{m}")\n                try:\n                    if pd.notna(v): cols[m]=float(v)\n                except Exception: pass\n            if not cols and self.player_index:\n                gid=str(row.get("game_id","")); pid=str(p).replace(".0","")\n                rec=self.player_index.get((gid,pid,side))\n                if rec is None:\n                    for (kg,kp,ks),rv in self.player_index.items():\n                        if kg==gid and ks==side and str(kp).replace(".0","")==pid: rec=rv; break\n                if rec:\n                    for m in ("era","whip","k9","bb9","hr9","fip","pitches","k_rate","bb_rate","ip","er","h","hr","bb","so"):\n                        try:\n                            v=rec.get(m)\n                            if v is not None and pd.notna(v): cols[m]=float(v)\n                        except Exception: pass\n            if cols:\n                cols["starts"]=1.0\n                self.pitcher_history[(row["league"],p)].append(cols)\n\n    # ------------------------------------------------------------------\n    # Models'''
s,n=pat.subn(lambda m:new,s,count=1)
if n!=1: raise RuntimeError('pitcher history target missing')
s='# BACKTEST_RUNTIME_HARDENING_V2\n'+s
P.write_text(s,encoding='utf-8')
print('[BACKTEST PATCH] V2 applied: NPB normalizer + starter metadata + player pitcher history')
