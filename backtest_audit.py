#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deterministic audit for the NPB collector/backtest pipeline."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data'; CP=DATA/'checkpoints'
def fail(msg): raise SystemExit('[AUDIT FAIL] '+msg)
def main():
    collector=(ROOT/'npb_multi_source.py').read_text(encoding='utf-8')
    backtest=(ROOT/'baseball_backtest.py').read_text(encoding='utf-8')
    npbpatch=(ROOT/'npb_runtime_patch.py').read_text(encoding='utf-8')
    btpatch=(ROOT/'baseball_backtest_runtime_patch.py').read_text(encoding='utf-8')
    if 'def _official_starters_from_npb' not in npbpatch: fail('official NPB starter runtime patch missing')
    if 'EMPTY SCHEDULE -> preserved checkpoint' not in npbpatch: fail('empty-schedule protection missing')
    if 'def _normalize_npb_pbp' not in btpatch: fail('NPB loader normalization patch missing')
    if 'home_starter_' not in btpatch or 'away_starter_' not in btpatch: fail('starter metadata propagation patch missing')
    if 'def _update_pitcher_history' not in backtest or 'self._update_pitcher_history(row)' not in backtest: fail('backtest history/update path missing')
    if backtest.find('match_features(row)') > backtest.find('self._update_pitcher_history(row)'): fail('pitcher history update appears before feature generation')
    if 'def aggregate_npb_games' not in backtest: fail('NPB aggregation missing')
    agg=DATA/'npb_multi_source_games_all.csv'
    if not agg.exists(): print('[AUDIT] code checks passed; aggregate data not present yet'); return
    d=pd.read_csv(agg,low_memory=False)
    if 'game_id' not in d: fail('aggregate has no game_id')
    dup=int(d.game_id.astype(str).duplicated().sum())
    if dup: fail(f'aggregate contains {dup} duplicate game IDs')
    if 'datetime' in d and pd.to_datetime(d.datetime,errors='coerce').isna().any(): fail('aggregate has invalid datetimes')
    print(f'[AUDIT] aggregate games={len(d)} unique_game_ids={d.game_id.nunique()}')
    status=CP/'npb_collection_status.json'
    if status.exists():
        s=json.loads(status.read_text(encoding='utf-8')); statuses=s.get('seasons_status',[])
        bad=[int(x.get('year')) for x in statuses if x.get('complete') and (x.get('unavailable') or float(x.get('coverage_pct',0))<70)]
        if bad: fail('invalid completed seasons: '+','.join(map(str,bad)))
        print(f'[AUDIT] collection_complete={s.get("complete")} unavailable_years={[int(x.get("year")) for x in statuses if x.get("unavailable")]}')
    print('[AUDIT PASS] static and data-integrity checks passed')
if __name__=='__main__': main()
