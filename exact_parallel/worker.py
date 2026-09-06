#!/usr/bin/env python3
"""Exact-parallel Baseball verification worker.

The canonical Baseball engine remains untouched. Each worker replays the
frozen chronological input with the exact production patch chain, then emits
only its deterministic output shard. No worker writes production state.
"""
from __future__ import annotations
import hashlib,json,os,subprocess,sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
W=int(os.environ["WORKER_ID"]); N=int(os.environ.get("WORKER_COUNT","4"))
OUT=ROOT/"exact_parallel"/"out"; OUT.mkdir(parents=True,exist_ok=True)

# Apply exactly the same hardening chain as production.
patches=["repair_npb_syntax.py","npb_runtime_patch.py","npb_official_schedule_patch.py","npb_quality_runtime_patch.py","baseball_backtest_runtime_patch.py","baseball_production_runtime_patch.py","baseball_mlb_score_hilo_patch.py","baseball_quality_runtime_patch.py"]
for patch in patches:
    subprocess.run([sys.executable,patch],cwd=ROOT,check=True)

# Never resume an old prediction checkpoint: this is a fresh replay against
# the frozen input snapshot.
import shutil
shutil.rmtree(ROOT/"results"/"checkpoints",ignore_errors=True)
for p in (ROOT/"results"/"npb_backtest_results.csv",ROOT/"results"/"mlb_backtest_results.csv",ROOT/"results"/"combined_backtest_results.csv"):
    p.unlink(missing_ok=True)

env=os.environ.copy(); env.update({"BASEBALL_TIME_BUDGET_SEC":"3600","MLB_ENRICH_STARTERS":"0"})
subprocess.run([sys.executable,"baseball_backtest.py","--npb-only","--data-dir","data"],cwd=ROOT,env=env,check=True)
subprocess.run([sys.executable,"baseball_backtest.py","--mlb-only","--data-dir","data","--mlb-start","2020","--mlb-end","2026"],cwd=ROOT,env=env,check=True)

files=[("NPB","results/npb_backtest_results.csv"),("MLB","results/mlb_backtest_results.csv")]
manifest={"worker":W,"count":N,"code_sha":os.environ.get("GITHUB_SHA","")}
for league,rel in files:
    p=ROOT/rel
    if not p.exists(): raise SystemExit(f"missing {rel}")
    d=pd.read_csv(p,low_memory=False)
    idx=np.array_split(np.arange(len(d)),N)[W]
    s=d.iloc[idx].copy(); s.insert(0,"__parallel_row",idx.astype(int)); s.to_csv(OUT/f"{league.lower()}_worker_{W}.csv",index=False)
    manifest[league]={"rows_total":len(d),"rows_shard":len(s),"sha256":hashlib.sha256(p.read_bytes()).hexdigest()}
(OUT/f"baseball_worker_{W}.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps(manifest,ensure_ascii=False))
