#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deterministic audit for the NPB collector/backtest pipeline."""
from __future__ import annotations
import json
import re
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data'
CP=DATA/'checkpoints'

def fail(msg):
    raise SystemExit('[AUDIT FAIL] '+msg)

def main():
    collector=(ROOT/'npb_multi_source.py').read_text(encoding='utf-8')
    backtest=(ROOT/'baseball_backtest.py').read_text(encoding='utf-8')
    if 'Never fall back to arbitrary pitcher ordering.' not in collector:
        fail('strict starter resolution is not active')
    if 'def _official_starters_from_npb' not in collector:
        fail('official NPB starter fallback is missing')
    if 'EMPTY SCHEDULE -> preserved checkpoint' not in collector:
        fail('empty schedule protection is missing')
    if 'def _update_pitcher_history' not in backtest or 'match_features(row)' not in backtest:
        fail('backtest history/update path missing')
    if backtest.find('match_features(row)') > backtest.find('self._update_pitcher_history(row)'):
        fail('pitcher history update appears before feature generation')

    agg=DATA/'npb_multi_source_games_all.csv'
    if not agg.exists():
        print('[AUDIT] collector code checks passed; aggregate data not present yet')
        return
    d=pd.read_csv(agg,low_memory=False)
    if 'game_id' not in d:
        fail('aggregate has no game_id')
    dup=int(d.game_id.astype(str).duplicated().sum())
    if dup:
        fail(f'aggregate contains {dup} duplicate game IDs')
    if 'datetime' in d:
        dt=pd.to_datetime(d.datetime,errors='coerce')
        if dt.isna().any():
            fail(f'aggregate has {int(dt.isna().sum())} invalid datetimes')
    print(f'[AUDIT] aggregate games={len(d)} unique_game_ids={d.game_id.nunique()}')

    status=CP/'npb_collection_status.json'
    if status.exists():
        s=json.loads(status.read_text(encoding='utf-8'))
        statuses=s.get('seasons_status',[])
        unavailable=[int(x.get('year')) for x in statuses if x.get('unavailable')]
        bad_complete=[int(x.get('year')) for x in statuses if x.get('complete') and (x.get('unavailable') or float(x.get('coverage_pct',0))<70)]
        if bad_complete:
            fail('invalid completed seasons: '+','.join(map(str,bad_complete)))
        print(f'[AUDIT] collection_complete={s.get("complete")} unavailable_years={unavailable}')
    print('[AUDIT PASS] static and data-integrity checks passed')

if __name__=='__main__':
    main()
