from __future__ import annotations

import pandas as pd


def evaluate(df: pd.DataFrame, table: str) -> tuple[pd.DataFrame, dict]:
    """Evaluate whether every source timestamp is available by the cutoff.

    The gate is fail-closed: missing columns, invalid timestamps, or any
    source timestamp after the prediction cutoff are failures.  The previous
    implementation derived the report status from the first row only, which
    could incorrectly PASS a mixed-validity dataset.
    """
    out = df.copy()

    if 'prediction_cutoff_at' not in out.columns:
        out['_cutoff_violation'] = False
        out['_cutoff_status'] = 'FAIL'
        return out, {
            'table': table,
            'status': 'FAIL',
            'reason': 'MISSING_PREDICTION_CUTOFF_AT',
            'violations': 0,
            'unverifiable_rows': len(out),
        }

    if 'source_timestamp' not in out.columns:
        out['_cutoff_violation'] = False
        out['_cutoff_status'] = 'FAIL'
        return out, {
            'table': table,
            'status': 'FAIL',
            'reason': 'MISSING_SOURCE_TIMESTAMP',
            'violations': 0,
            'unverifiable_rows': len(out),
        }

    cutoff = pd.to_datetime(out['prediction_cutoff_at'], errors='coerce', utc=True)
    source = pd.to_datetime(out['source_timestamp'], errors='coerce', utc=True)
    invalid_time = cutoff.isna() | source.isna()
    violation = (~invalid_time) & (source > cutoff)

    # Any unverifiable row is a hard failure under the gate policy.
    fail_mask = invalid_time | violation
    out['_cutoff_violation'] = violation
    out['_cutoff_unverifiable'] = invalid_time
    out['_cutoff_status'] = 'FAIL' if fail_mask.any() else ('PASS' if len(out) else 'EMPTY')

    return out, {
        'table': table,
        'status': 'FAIL' if fail_mask.any() else ('PASS' if len(out) else 'EMPTY'),
        'reason': (
            'INVALID_OR_MISSING_TIMESTAMP' if invalid_time.any()
            else ('SOURCE_AFTER_CUTOFF' if violation.any() else 'OK')
        ),
        'violations': int(violation.sum()),
        'unverifiable_rows': int(invalid_time.sum()),
        'failed_rows': int(fail_mask.sum()),
    }
