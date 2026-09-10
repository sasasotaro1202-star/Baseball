from __future__ import annotations

import pandas as pd


def evaluate(df: pd.DataFrame, table: str) -> tuple[pd.DataFrame, dict]:
    out = df.copy()
    if 'prediction_cutoff_at' not in out.columns:
        out['_cutoff_status'] = 'UNVERIFIABLE_NO_CUTOFF'
        return out, {'table': table, 'status': 'UNVERIFIABLE_NO_CUTOFF', 'violations': 0}
    if 'source_timestamp' not in out.columns:
        out['_cutoff_status'] = 'UNVERIFIABLE_NO_SOURCE_TIMESTAMP'
        return out, {'table': table, 'status': 'UNVERIFIABLE_NO_SOURCE_TIMESTAMP', 'violations': 0}
    cutoff = pd.to_datetime(out['prediction_cutoff_at'], errors='coerce', utc=True)
    source = pd.to_datetime(out['source_timestamp'], errors='coerce', utc=True)
    invalid_time = cutoff.isna() | source.isna()
    violation = (~invalid_time) & (source > cutoff)
    out['_cutoff_violation'] = violation
    out['_cutoff_status'] = 'FAIL' if violation.any() else ('UNVERIFIABLE' if invalid_time.any() else 'PASS')
    return out, {
        'table': table,
        'status': out['_cutoff_status'].iloc[0] if len(out) else 'EMPTY',
        'violations': int(violation.sum()),
        'unverifiable_rows': int(invalid_time.sum()),
    }
