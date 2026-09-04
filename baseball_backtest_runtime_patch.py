#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime hardening for the NPB/MLB backtest engine.

The checked-in backtest is kept intact; this patch adds compatibility with the
multi-source collector's game-level starter IDs and player-game micro-features.
All player performance is still consumed only after the target game is scored.
"""
from __future__ import annotations
import re
from pathlib import Path

P=Path('baseball_backtest.py')
s=P.read_text(encoding='utf-8')
if '# BACKTEST_RUNTIME_HARDENING_V1' in s:
    print('[BACKTEST PATCH] V1 already applied')
    raise SystemExit(0)

old='''            hp = self._first_pitcher(g, "home")
            ap = self._first_pitcher(g, "away")
            rows.append({'''
new='''            hp = self._first_value(g, "home_starter") or self._first_pitcher(g, "home")
            ap = self._first_value(g, "away_starter") or self._first_pitcher(g, "away")
            rows.append({'''
if old not in s: raise RuntimeError('starter aggregation target not found')
s=s.replace(old,new,1)

anchor='''    def _first_pitcher(self, g: pd.DataFrame, side: str) -> str:
'''
helper='''    def _first_value(self, g: pd.DataFrame, col: str) -> str:
        if col not in g: return ""
        for x in g[col].tolist():
            v=str(x).strip()
            if v and v.lower() not in ("nan","none"):
                return v
        return ""

'''
if anchor not in s: raise RuntimeError('first pitcher anchor not found')
s=s.replace(anchor,helper+anchor,1)

pat=re.compile(r"    def _lineup_ids\(self, row: pd.Series, side: str\) -> List\[Dict\[str, Any\]\]:.*?\n\n    def _player_profile",re.S)
new='''    def _lineup_ids(self, row: pd.Series, side: str) -> List[Dict[str, Any]]:
        raw=row.get(f'{side}_lineup_json','')
        if raw not in (None,'','nan'):
            try:
                x=json.loads(raw) if isinstance(raw,str) else raw
                if isinstance(x,list) and x: return x
            except Exception:
                pass
        # Collector compatibility: player-game rows contain batting_order and
        # player_id for the target game's lineup. We use only identity/order
        # metadata here; performance is obtained from prior-game history.
        if not self.player_game.empty and 'game_id' in self.player_game.columns:
            gid=str(row.get('game_id',''))
            d=self.player_game[self.player_game.game_id.astype(str)==gid].copy()
            if 'side' in d.columns:
                d=d[d.side.astype(str).str.lower()==side.lower()]
            if 'role' in d.columns:
                d=d[~d.role.astype(str).str.lower().eq('pitcher')]
            if not d.empty and 'batting_order' in d.columns and 'player_id' in d.columns:
                d['bo']=pd.to_numeric(d['batting_order'],errors='coerce')
                d=d[d.bo.notna()].sort_values(['bo','player_id']).drop_duplicates('player_id')
                return [{'player_id':str(r.player_id),'batting_order':float(r.bo),
                         'player_name':str(r.get('player_name','') or ''),
                         'position':str(r.get('position','') or ''),
                         'hand':str(r.get('hand','') or '')} for _,r in d.iterrows()]
        return []

    def _player_profile'''
s,n=pat.subn(lambda m:new,s,count=1)
if n!=1: raise RuntimeError('lineup target not found')

pat=re.compile(r"    def _update_pitcher_history\(self, row: pd.Series\):.*?\n\n    # ------------------------------------------------------------------\n    # Models",re.S)
new='''    def _update_pitcher_history(self, row: pd.Series):
        for side in ("home", "away"):
            p = str(row.get(f"{side}_starter", "") or "").strip()
            if not p or p.lower() in ("nan","none"):
                continue
            metrics = row.get(f"{side}_starter_metrics")
            if isinstance(metrics, dict):
                self.pitcher_history[(row["league"], p)].append(metrics)
                continue
            cols = {}
            for m in ("era", "whip", "k9", "bb9", "hr9", "fip", "pitches", "k_rate", "bb_rate", "ip", "er", "h", "hr", "bb", "so"):
                v = row.get(f"{side}_starter_{m}")
                if pd.notna(v):
                    try: cols[m] = float(v)
                    except Exception: pass
            if not cols and self.player_index:
                gid=str(row.get('game_id',''))
                pid=str(p).replace('.0','')
                key=(gid,pid,side)
                rec=self.player_index.get(key)
                if rec is None:
                    for (kg,kp,ks),rv in self.player_index.items():
                        if kg==gid and ks==side and str(kp).replace('.0','')==pid:
                            rec=rv; break
                if rec is not None:
                    for m in ("era","whip","k9","bb9","hr9","fip","pitches","k_rate","bb_rate","ip","er","h","hr","bb","so"):
                        v=rec.get(m)
                        try:
                            if v is not None and pd.notna(v) and float(v)==float(v): cols[m]=float(v)
                        except Exception: pass
            if cols:
                cols["starts"] = 1.0
                self.pitcher_history[(row["league"], p)].append(cols)

    # ------------------------------------------------------------------
    # Models'''
s,n=pat.subn(lambda m:new,s,count=1)
if n!=1: raise RuntimeError('pitcher history target not found')

s='# BACKTEST_RUNTIME_HARDENING_V1\n'+s
P.write_text(s,encoding='utf-8')
print('[BACKTEST PATCH] V1 applied: starter IDs + player lineup metadata + pitcher micro-feature history')
