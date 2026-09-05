#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deterministic integrity/leakage audit for the production NPB/MLB pipeline."""
from __future__ import annotations
import ast,json,re
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parent; DATA=ROOT/"data"; CP=DATA/"checkpoints"
REQUIRED=(".github/workflows/baseball_production.yml",".github/workflows/baseball_audit.yml","baseball_backtest.py","baseball_backtest_runtime_patch.py","baseball_production_runtime_patch.py","baseball_mlb_score_hilo_patch.py","baseball_quality_runtime_patch.py","npb_multi_source.py","npb_runtime_patch.py","npb_official_schedule_patch.py","npb_quality_runtime_patch.py","source_quality_gate.py","DATA_SOURCE_POLICY.json")
def fail(msg): raise SystemExit("[AUDIT FAIL] "+msg)
def assert_ast_function(text,filename,name):
    try: tree=ast.parse(text,filename=filename)
    except SyntaxError as e: fail(f"syntax error in {filename}: {e}")
    if not any(isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==name for n in ast.walk(tree)): fail(f"{filename}: function {name} missing")
def main():
    for rel in REQUIRED:
        if not (ROOT/rel).exists(): fail(f"required file missing: {rel}")
    collector=(ROOT/"npb_multi_source.py").read_text(encoding="utf-8"); backtest=(ROOT/"baseball_backtest.py").read_text(encoding="utf-8"); npbpatch=(ROOT/"npb_runtime_patch.py").read_text(encoding="utf-8"); official=(ROOT/"npb_official_schedule_patch.py").read_text(encoding="utf-8"); npbq=(ROOT/"npb_quality_runtime_patch.py").read_text(encoding="utf-8"); btpatch=(ROOT/"baseball_backtest_runtime_patch.py").read_text(encoding="utf-8"); prod=(ROOT/"baseball_production_runtime_patch.py").read_text(encoding="utf-8"); mlb=(ROOT/"baseball_mlb_score_hilo_patch.py").read_text(encoding="utf-8"); quality=(ROOT/"baseball_quality_runtime_patch.py").read_text(encoding="utf-8"); workflow=(ROOT/".github/workflows/baseball_production.yml").read_text(encoding="utf-8"); policy=json.loads((ROOT/"DATA_SOURCE_POLICY.json").read_text(encoding="utf-8"))
    checks=[("RUNTIME_HARDENING_V5",npbpatch),("BACKTEST_RUNTIME_HARDENING_V4",btpatch),("BASEBALL_PRODUCTION_HARDENING_V1",prod),("MLB_SCORE_HILO_PATCH_V2",mlb),("BASEBALL_QUALITY_HARDENING_V1",quality),("NPB_QUALITY_HARDENING_V1",npbq),("OFFICIAL_NPB_SCHEDULE_VALIDATION_V1",official)]
    for marker,text in checks:
        if marker not in text: fail(f"required hardening marker missing: {marker}")
    for needle in ("pred_score1","pred_score4","pred_low_prob","pred_high_prob","actual_low_high","score_exact_hit","low_high_hit"):
        if needle not in mlb: fail(f"MLB internal prediction contract missing: {needle}")
    if "confirmed_starters" not in mlb or "starter_rate < 0.90" not in mlb: fail("MLB confirmed-starter gate missing")
    if "weather_forecast_asof_cutoff" not in mlb: fail("MLB observed-weather leakage guard missing")
    if "_official_starters_from_npb" not in npbpatch or "Strict rule: unresolved starters remain unresolved" not in npbpatch: fail("NPB strict official starter resolver missing")
    if "EMPTY SCHEDULE -> preserved checkpoint" not in npbq: fail("NPB empty-schedule checkpoint protection missing")
    if "repair_or_missing" not in npbq: fail("NPB stale-checkpoint repair missing")
    if "_official_validate_games" not in official or "schedule_{month:02d}_detail.html" not in official: fail("official NPB monthly schedule validation missing")
    if not all(x in prod for x in ("npb_official_schedule_patch.py","npb_quality_runtime_patch.py","baseball_mlb_score_hilo_patch.py","baseball_quality_runtime_patch.py")): fail("production hardening chain incomplete")
    if "_normalize_npb_pbp" not in btpatch or "home_starter_" not in btpatch or "away_starter_" not in btpatch: fail("NPB normalization/starter propagation missing")
    if "Asia/Tokyo" not in btpatch: fail("naive NPB datetimes are not explicitly interpreted as JST")
    for needle in ("npb_runtime_patch.py","baseball_backtest_runtime_patch.py","baseball_production_runtime_patch.py","baseball_backtest.py","source_quality_gate.py","MLB_ENRICH_STARTERS: \"1\"","baseball_quality_runtime_patch.py","npb_quality_runtime_patch.py"):
        if needle not in workflow: fail(f"production workflow missing required reference: {needle}")
    if not re.search(r"NPB_MIN_STARTER_LINE_COVERAGE\s*:\s*['\"]?70(?:\.0)?['\"]?",workflow): fail("NPB starter threshold missing")
    if not re.search(r"MLB_MIN_STARTER_COVERAGE\s*:\s*['\"]?90(?:\.0)?['\"]?",workflow): fail("MLB starter threshold missing")
    if not re.search(r"NPB_COLLECTION_BUDGET_SEC\s*:\s*['\"]?3600['\"]?",workflow): fail("NPB 60-minute collection budget missing")
    if not re.search(r"BASEBALL_TIME_BUDGET_SEC\s*:\s*['\"]?3600['\"]?",workflow): fail("NPB 60-minute backtest budget missing")
    if not re.search(r"timeout-minutes:\s*70",workflow): fail("production timeout missing")
    if not re.search(r"BASEBALL_TIME_BUDGET_SEC:\s*\"3600\"",workflow): fail("MLB quality-first runtime budget missing")
    if "if: always()" not in workflow or "actions/download-artifact@v4" not in workflow: fail("artifact recovery missing")
    if "npb.jp" not in json.dumps(policy.get("NPB",{}),ensure_ascii=False).lower(): fail("NPB policy does not name official NPB")
    if "statsapi.mlb.com" not in json.dumps(policy.get("MLB",{}),ensure_ascii=False).lower(): fail("MLB policy does not name MLB Stats API")
    for name,text in (("baseball_backtest.py",backtest),("npb_multi_source.py",collector),("npb_runtime_patch.py",npbpatch),("npb_official_schedule_patch.py",official),("npb_quality_runtime_patch.py",npbq),("baseball_backtest_runtime_patch.py",btpatch),("baseball_production_runtime_patch.py",prod),("baseball_mlb_score_hilo_patch.py",mlb),("baseball_quality_runtime_patch.py",quality)):
        try: ast.parse(text,filename=name)
        except SyntaxError as e: fail(f"syntax error in {name}: {e}")
    for name in ("_update_pitcher_history","match_features","aggregate_npb_games","fit_score_ensemble","predict_scores"): assert_ast_function(backtest,"baseball_backtest.py",name)
    feature_pos=backtest.find("match_features(row)"); update_pos=backtest.find("self._update_pitcher_history(row)")
    if feature_pos<0 or update_pos<0: fail("prediction/history call path missing")
    if feature_pos>update_pos: fail("pitcher history is updated before target feature generation")
    effective_score = backtest + "\n" + btpatch
    for needle in ("_score_prior","prior_blend","_nb_nll","dispersion_home","dispersion_away"):
        if needle not in effective_score: fail(f"adaptive score layer missing: {needle}")
    agg=DATA/"npb_multi_source_games_all.csv"
    if not agg.exists(): print("[AUDIT] code/leakage-order/workflow/source-policy checks passed; aggregate data not present yet"); print("[AUDIT PASS]"); return
    d=pd.read_csv(agg,low_memory=False)
    if "game_id" not in d: fail("aggregate has no game_id")
    ids=d.game_id.astype(str).str.replace(r"\.0$","",regex=True); dup=int(ids.duplicated().sum())
    if dup: fail(f"aggregate contains {dup} duplicate game IDs")
    if "datetime" not in d: fail("aggregate has no datetime")
    dt=pd.to_datetime(d.datetime,errors="coerce",utc=True)
    if dt.isna().any(): fail(f"aggregate has {int(dt.isna().sum())} invalid datetimes")
    print(f"[AUDIT] aggregate games={len(d)} unique_game_ids={ids.nunique()}")
    print("[AUDIT PASS] static, leakage-order, workflow, source-policy, and data-integrity checks passed")
if __name__=="__main__": main()
