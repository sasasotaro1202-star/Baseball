#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deterministic integrity/leakage audit for the NPB collector/backtest pipeline."""
from __future__ import annotations
import ast
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
CP = DATA / "checkpoints"
REQUIRED = (
    ".github/workflows/baseball_backtest.yml",
    ".github/workflows/baseball_audit.yml",
    "baseball_backtest.py",
    "baseball_backtest_runtime_patch.py",
    "npb_multi_source.py",
    "npb_runtime_patch.py",
)

def fail(msg: str) -> None:
    raise SystemExit("[AUDIT FAIL] " + msg)

def assert_ast_function(text: str, filename: str, name: str) -> ast.FunctionDef:
    try:
        tree = ast.parse(text, filename=filename)
    except SyntaxError as e:
        fail(f"syntax error in {filename}: {e}")
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    fail(f"{filename}: function {name} missing")

def main() -> None:
    for rel in REQUIRED:
        if not (ROOT / rel).exists():
            fail(f"required file missing: {rel}")

    collector = (ROOT / "npb_multi_source.py").read_text(encoding="utf-8")
    backtest = (ROOT / "baseball_backtest.py").read_text(encoding="utf-8")
    npbpatch = (ROOT / "npb_runtime_patch.py").read_text(encoding="utf-8")
    btpatch = (ROOT / "baseball_backtest_runtime_patch.py").read_text(encoding="utf-8")
    workflow = (ROOT / ".github/workflows/baseball_backtest.yml").read_text(encoding="utf-8")

    # Runtime hardening must remain active and idempotent.
    if "RUNTIME_HARDENING_V4" not in npbpatch:
        fail("NPB runtime hardening V4 missing")
    if "BACKTEST_RUNTIME_HARDENING_V2" not in btpatch:
        fail("backtest runtime hardening V2 missing")
    if "_official_starters_from_npb" not in npbpatch:
        fail("official NPB starter resolver missing")
    if "Strict rule: unresolved starters remain unresolved" not in npbpatch:
        fail("starter resolver still permits guessing")
    if "EMPTY SCHEDULE -> preserved checkpoint" not in npbpatch:
        fail("empty-schedule checkpoint protection missing")
    if "MIN_STARTER_LINE_COVERAGE" not in npbpatch:
        fail("starter coverage gate missing")
    if "_normalize_npb_pbp" not in btpatch:
        fail("NPB loader normalization missing")
    if "home_starter_" not in btpatch or "away_starter_" not in btpatch:
        fail("starter metric propagation missing")

    # Parse every production source before trusting string-level invariants.
    for name, text in (("baseball_backtest.py", backtest), ("npb_multi_source.py", collector),
                       ("npb_runtime_patch.py", npbpatch), ("baseball_backtest_runtime_patch.py", btpatch)):
        try:
            ast.parse(text, filename=name)
        except SyntaxError as e:
            fail(f"syntax error in {name}: {e}")

    assert_ast_function(backtest, "baseball_backtest.py", "_update_pitcher_history")
    assert_ast_function(backtest, "baseball_backtest.py", "match_features")
    assert_ast_function(backtest, "baseball_backtest.py", "aggregate_npb_games")

    # Target-game leakage guard: feature generation must occur before history mutation.
    feature_pos = backtest.find("match_features(row)")
    update_pos = backtest.find("self._update_pitcher_history(row)")
    if feature_pos < 0 or update_pos < 0:
        fail("prediction/history call path missing")
    if feature_pos > update_pos:
        fail("pitcher history is updated before target feature generation")

    # Workflow must compile patches and run the actual backtest after collection.
    for needle in ("npb_runtime_patch.py", "baseball_backtest_runtime_patch.py", "backtest_audit.py", "baseball_backtest.py"):
        if needle not in workflow:
            fail(f"workflow missing required step/reference: {needle}")
    if "NPB_MIN_STARTER_LINE_COVERAGE=70" not in workflow:
        fail("workflow starter coverage threshold missing")

    agg = DATA / "npb_multi_source_games_all.csv"
    if not agg.exists():
        print("[AUDIT] code/leakage checks passed; aggregate data not present yet")
        print("[AUDIT PASS]")
        return

    d = pd.read_csv(agg, low_memory=False)
    if "game_id" not in d.columns:
        fail("aggregate has no game_id")
    ids = d["game_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    dup = int(ids.duplicated().sum())
    if dup:
        fail(f"aggregate contains {dup} duplicate game IDs")
    if "datetime" not in d.columns:
        fail("aggregate has no datetime")

    dt = pd.to_datetime(d["datetime"], errors="coerce", utc=True)
    if dt.isna().any():
        fail(f"aggregate has {int(dt.isna().sum())} invalid datetimes")
    if not dt.is_monotonic_increasing:
        print("[AUDIT] aggregate is not globally sorted; sorting is required before walk-forward evaluation")

    # A game cannot contribute postgame observations to its own pregame feature set.
    if "date" in d.columns:
        date_dt = pd.to_datetime(d["date"], errors="coerce", utc=True)
        if date_dt.isna().any():
            fail("aggregate contains invalid date values")
        if (date_dt > dt + pd.Timedelta(days=1)).any() or (dt > date_dt + pd.Timedelta(days=1)).any():
            fail("aggregate datetime/date fields disagree by more than one day")

    print(f"[AUDIT] aggregate games={len(d)} unique_game_ids={ids.nunique()}")

    status = CP / "npb_collection_status.json"
    if status.exists():
        s = json.loads(status.read_text(encoding="utf-8"))
        statuses = s.get("seasons_status", [])
        if not isinstance(statuses, list):
            fail("collection status seasons_status is not a list")
        bad = []
        for x in statuses:
            try:
                if x.get("complete") and (x.get("unavailable") or float(x.get("coverage_pct", 0)) < 70):
                    bad.append(int(x.get("year")))
            except Exception:
                fail("malformed season status entry")
        if bad:
            fail("invalid completed seasons: " + ",".join(map(str, bad)))
        print(f"[AUDIT] collection_complete={s.get('complete')} unavailable_years={[int(x.get('year')) for x in statuses if x.get('unavailable')]}")

    # Enriched data must contain explicit starter confirmation fields if present.
    if "home_starter" in d.columns and "away_starter" in d.columns:
        both = (d["home_starter"].fillna("").astype(str).str.strip() != "") & (d["away_starter"].fillna("").astype(str).str.strip() != "")
        print(f"[AUDIT] both_starter_rows={int(both.sum())}/{len(d)} coverage_pct={100.0*float(both.mean()):.1f}")

    print("[AUDIT PASS] static, leakage-order, workflow, and data-integrity checks passed")

if __name__ == "__main__":
    main()
