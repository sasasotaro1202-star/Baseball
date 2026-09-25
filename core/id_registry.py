from __future__ import annotations

import hashlib
import re
from typing import Iterable

import pandas as pd


def token(value: object) -> str:
    s = '' if value is None else str(value).strip().casefold()
    s = re.sub(r'[^\w]+', '_', s, flags=re.UNICODE).strip('_')
    return s


def stable_id(prefix: str, parts: Iterable[object]) -> str:
    payload = '|'.join(token(x) for x in parts)
    return f'{prefix}_{hashlib.sha256(payload.encode()).hexdigest()[:20]}'


def integrate_game_ids(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if 'game_id' not in out:
        out['game_id'] = pd.NA
    missing = out['game_id'].isna() | out['game_id'].astype(str).str.strip().eq('')
    out['game_id_source'] = 'source'
    for idx in out.index[missing]:
        row = out.loc[idx]
        out.at[idx, 'game_id'] = stable_id(
            'game',
            [row.get('sport'), row.get('league'), row.get('date'),
             row.get('home_team'), row.get('away_team'), row.get('venue')],
        )
        out.at[idx, 'game_id_source'] = 'derived_stable_identity'
    if 'home_team' in out.columns:
        out['home_team_id'] = out['home_team'].map(
            lambda x: stable_id('team', [x]) if pd.notna(x) else pd.NA
        )
    if 'away_team' in out.columns:
        out['away_team_id'] = out['away_team'].map(
            lambda x: stable_id('team', [x]) if pd.notna(x) else pd.NA
        )
    return out


def integrate_entity_ids(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if 'team' in out.columns:
        out['team_id'] = out['team'].map(
            lambda x: stable_id('team', [x]) if pd.notna(x) else pd.NA
        )
    if 'player' in out.columns:
        out['player_id'] = out['player'].map(
            lambda x: stable_id('player', [x]) if pd.notna(x) else pd.NA
        )
    return out
