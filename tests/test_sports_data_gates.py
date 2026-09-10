from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from core.cutoff_gate import evaluate
from core.id_registry import integrate_game_ids, integrate_entity_ids
from core.schema_gate import gate

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = json.loads((ROOT / 'schemas/complete_sports_data_schema.json').read_text())


def test_schema_gate_fails_closed_on_missing_columns():
    result = gate(pd.DataFrame({'sport': ['MLB']}), 'games', SCHEMA)
    assert result['pass'] is False
    assert 'game_id' in result['missing_columns']


def test_identity_is_deterministic():
    df = pd.DataFrame([{
        'sport': 'MLB', 'league': 'MLB', 'game_id': None,
        'date': '2026-09-10', 'home_team': 'A', 'away_team': 'B', 'venue': 'X'
    }])
    a = integrate_game_ids(df.copy())
    b = integrate_game_ids(df.copy())
    assert a.loc[0, 'game_id'] == b.loc[0, 'game_id']
    assert a.loc[0, 'home_team_id'] == b.loc[0, 'home_team_id']


def test_cutoff_gate_rejects_future_source_timestamp():
    df = pd.DataFrame([{
        'prediction_cutoff_at': '2026-09-10T12:00:00Z',
        'source_timestamp': '2026-09-10T12:01:00Z'
    }])
    checked, report = evaluate(df, 'games')
    assert report['status'] == 'FAIL'
    assert int(checked['_cutoff_violation'].sum()) == 1


def test_cutoff_gate_rejects_missing_timestamps():
    df = pd.DataFrame([{'prediction_cutoff_at': None, 'source_timestamp': None}])
    _, report = evaluate(df, 'games')
    assert report['status'].startswith('UNVERIFIABLE')
