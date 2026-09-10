from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from core.schema_gate import load_schema, gate
from core.id_registry import integrate_game_ids, integrate_entity_ids
from core.cutoff_gate import evaluate

TABLE_PATTERNS = {
    'games': ['*games*.csv', '*matches*.csv', '*fixtures*.csv', '*football*.csv', '*soccer*.csv'],
    'pitching': ['*pitch*.csv', '*starter*.csv', '*probable*.csv'],
    'lineups': ['*lineup*.csv', '*starting_*.csv'],
    'bullpen_usage': ['*bullpen*.csv', '*reliever*.csv'],
    'statcast': ['*statcast*.csv', '*batted_ball*.csv'],
    'team_stats': ['*team_stats*.csv', '*standings*.csv', '*xg*.csv'],
    'player_stats': ['*player_stats*.csv', '*batting*.csv', '*players*.csv'],
    'events': ['*events*.csv', '*event*.csv', '*play_by_play*.csv'],
    'weather_venue': ['*weather*.csv', '*venue*.csv', '*park*.csv'],
    'odds': ['*odds*.csv', '*moneyline*.csv', '*betting*.csv'],
}

ALIASES = {
    'date': ['date', 'gamedate', 'game_date', 'datetime', 'game_datetime'],
    'game_id': ['game_id', 'gameid', 'match_id', 'matchid', 'id'],
    'league': ['league', 'competition', 'tournament'],
    'home_team': ['home_team', 'hometeam', 'home', 'home_team_name'],
    'away_team': ['away_team', 'awayteam', 'away', 'away_team_name'],
    'home_score': ['home_score', 'homescore', 'home_goals', 'fthg'],
    'away_score': ['away_score', 'awayscore', 'away_goals', 'ftag'],
}


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [re.sub(r'[^a-z0-9]+', '_', str(c).strip().lower()).strip('_') for c in out.columns]
    return out


def apply_aliases(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for target, candidates in ALIASES.items():
        if target not in out.columns:
            src = next((c for c in candidates if c in out.columns), None)
            if src:
                out[target] = out[src]
    if 'date' in out.columns:
        out['date'] = pd.to_datetime(out['date'], errors='coerce', utc=True)
    for col in ('home_score', 'away_score'):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors='coerce')
    return out


def load_table(input_dir: Path, patterns: list[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for pattern in patterns:
        for path in sorted(input_dir.glob(pattern)):
            try:
                df = clean_columns(pd.read_csv(path))
                df['source_file'] = path.name
                # This is ingestion time only. It is deliberately NOT used as proof
                # that the source was available before prediction_cutoff_at.
                df['ingested_at'] = pd.Timestamp.now(tz='UTC')
                frames.append(df)
            except Exception as exc:
                print(f'[READ-SKIP] {path}: {exc}')
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='/content/sports_prediction/input')
    parser.add_argument('--output', default='/content/sports_prediction/output')
    parser.add_argument('--schema', default='schemas/complete_sports_data_schema.json')
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    schema = load_schema(Path(args.schema))

    schema_reports: list[dict] = []
    cutoff_reports: list[dict] = []
    coverage: list[dict] = []

    for table, patterns in TABLE_PATTERNS.items():
        df = load_table(input_dir, patterns)
        if table == 'games':
            df = apply_aliases(df)
        if not df.empty:
            if table == 'games':
                df = integrate_game_ids(df)
            df = integrate_entity_ids(df)

        schema_report = gate(df, table, schema)
        schema_reports.append(schema_report)
        if not schema_report['pass']:
            print(f'[SCHEMA-FAIL] {table}: missing {schema_report["missing_columns"]}')
            continue

        # Never invent a historical prediction cutoff. A real source timestamp and
        # a real prediction_cutoff_at are required before a row can be used safely.
        checked, cutoff_report = evaluate(df, table)
        cutoff_reports.append(cutoff_report)
        coverage.append({
            'table': table,
            'rows': len(df),
            'files': int(df['source_file'].nunique()) if 'source_file' in df else 0,
        })
        checked.to_csv(output_dir / f'{table}_normalized.csv', index=False)

    pd.DataFrame(schema_reports).to_csv(output_dir / 'schema_gate_report.csv', index=False)
    pd.DataFrame(cutoff_reports).to_csv(output_dir / 'cutoff_gate_report.csv', index=False)
    pd.DataFrame(coverage).to_csv(output_dir / 'coverage_report.csv', index=False)

    schema_failures = [r for r in schema_reports if not r['pass']]
    cutoff_failures = [r for r in cutoff_reports if r['status'] == 'FAIL']
    cutoff_unverifiable = [r for r in cutoff_reports if str(r['status']).startswith('UNVERIFIABLE')]
    summary = {
        'schema_pass': not schema_failures,
        'cutoff_pass': not cutoff_failures and not cutoff_unverifiable,
        'schema_failures': schema_failures,
        'cutoff_failures': cutoff_failures,
        'cutoff_unverifiable': cutoff_unverifiable,
        'no_fabrication': True,
    }
    (output_dir / 'gate_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding='utf-8',
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))

    if schema_failures or cutoff_failures or cutoff_unverifiable:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
