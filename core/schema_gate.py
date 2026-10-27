from __future__ import annotations

import json
from pathlib import Path
import pandas as pd


def load_schema(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def gate(df: pd.DataFrame, table: str, schema: dict) -> dict:
    required = list(schema['tables'][table])
    missing = [c for c in required if c not in df.columns]
    return {
        'table': table,
        'required_columns': len(required),
        'missing_columns': '|'.join(missing),
        'pass': not missing,
    }
