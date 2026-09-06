#!/usr/bin/env python3
"""Strict equality gate for four exact-parallel Baseball workers."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
for league in ("npb","mlb"):
    base=ROOT/"exact_parallel"/"baseline"/f"{league}_backtest_results.csv"
    workers=[ROOT/"exact_parallel"/"out"/f"{league}_worker_{i}.csv" for i in range(4)]
    if not base.exists(): raise SystemExit(f"missing baseline {league}")
    if not all(p.exists() for p in workers): raise SystemExit(f"missing worker {league}")
    b=pd.read_csv(base,low_memory=False); a=pd.concat([pd.read_csv(p,low_memory=False) for p in workers],ignore_index=True)
    if len(a)!=len(b): raise SystemExit(f"{league}: row count mismatch baseline={len(b)} merged={len(a)}")
    if a["__parallel_row"].duplicated().any() or set(a["__parallel_row"])!=set(range(len(b))): raise SystemExit(f"{league}: shard gap/overlap")
    a=a.sort_values("__parallel_row",kind="mergesort").drop(columns=["__parallel_row"]).reset_index(drop=True)
    if list(a.columns)!=list(b.columns): raise SystemExit(f"{league}: schema mismatch")
    for c in b.columns:
        if pd.api.types.is_numeric_dtype(b[c]):
            x=a[c].to_numpy(float); y=b[c].to_numpy(float)
            if not np.array_equal(x,y,equal_nan=True):
                d=np.nanmax(np.abs(x-y)); raise SystemExit(f"{league}: numeric mismatch {c} max_abs={d}")
        elif not a[c].astype(object).equals(b[c].astype(object)):
            raise SystemExit(f"{league}: column mismatch {c}")
    out=ROOT/"exact_parallel"/f"merged_{league}_backtest_results.csv"; a.to_csv(out,index=False)
(Path(ROOT/"exact_parallel"/"equality_gate.json")).write_text(json.dumps({"status":"PASS","workers":4,"message":"NPB and MLB four-worker outputs are exactly identical to the single-run baseline."},ensure_ascii=False,indent=2),encoding="utf-8")
print(Path(ROOT/"exact_parallel"/"equality_gate.json").read_text())
