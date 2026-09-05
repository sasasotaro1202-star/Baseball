#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deterministic integrity/leakage audit for the production NPB/MLB pipeline."""
from __future__ import annotations
import ast
import json
import re
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
CP = DATA / "checkpoints"
REQUIRED = (
    ".github/workflows/baseball_production.yml",
    ".github/workflows/baseball_audit.yml",
    "baseball_backtest.py",
    "baseball_backtest_runtime_patch.py",
    "baseball_production_runtime_patch.py",
    "npb_multi_source.py",
    "npb_runtime_patch.py",
    "npb_official_schedule_patch.py",
    "source_quality_gate.py",
    "DATA_SOURCE_POLICY.json",
)


def fail(msg: str) -> None:
    raise SystemExit("[AUDIT FAIL] " + msg)


def assert_ast_function(text: str, filename: str, name: str) -> None:
    try:
        tree = ast.parse(text, filename=filename)
    except SyntaxError as e:
        fail(f"syntax error in {filename}: {e}")
    if not any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name for n in ast.walk(tree)):
        fail(f"{filename}: function {name} missing")


def main() -> None:
    for rel in REQUIRED:
        if not (ROOT / rel).exists():
            fail(f"required file missing: {rel}")

    collector = (ROOT / "npb_multi_source.py").read_text(encoding="utf-8")
    backtest = (ROOT / "baseball_backtest.py").read_text(encoding="utf-8")
    npbpatch = (ROOT / "npb_runtime_patch.py").read_text(encoding="utf-8")
    officialpatch = (ROOT / "npb_official_schedule_patch.py").read_text(encoding="utf-8")
    btpatch = (ROOT / "baseball_backtest_runtime_patch.py").read_text(encoding="utf-8")
    prodpatch = (ROOT / "baseball_production_runtime_patch.py").read_text(encoding="utf-8")
    workflow = (ROOT / ".github/workflows/baseball_production.yml").read_text(encoding="utf-8")
    policy = json.loads((ROOT / "DATA_SOURCE_POLICY.json").read_text(encoding="utf-8"))

    if "RUNTIME_HARDENING_V4" not in npbpatch:
        fail("NPB runtime hardening V4 missing")
    if "BACKTEST_RUNTIME_HARDENING_V3" not in btpatch:
        fail("backtest runtime hardening V3 missing")
    if "BASEBALL_PRODUCTION_HARDENING_V1" not in prodpatch:
        fail("production hardening V1 missing")
    if "_official_starters_from_npb" not in npbpatch:
        fail("official NPB starter resolver missing")
    if "Strict rule: unresolved starters remain unresolved" not in npbpatch:
        fail("starter resolver still permits guessing")
    if "EMPTY SCHEDULE -> preserved checkpoint" not in npbpatch:
        fail("empty-schedule checkpoint protection missing")
    if "MIN_STARTER_LINE_COVERAGE" not in npbpatch:
        fail("starter coverage gate missing")
    if "OFFICIAL_NPB_SCHEDULE_VALIDATION_V1" not in officialpatch:
        fail("official NPB schedule/result validation patch missing")
    if "_official_validate_games" not in officialpatch or "schedule_{month:02d}_detail.html" not in officialpatch:
        fail("official NPB monthly schedule validation implementation missing")
    if "npb_official_schedule_patch.py" not in prodpatch:
        fail("production hardening does not chain official NPB validation")
    if "_normalize_npb_pbp" not in btpatch:
        fail("NPB loader normalization missing")
    if "home_starter_" not in btpatch or "away_starter_" not in btpatch:
        fail("starter metric propagation missing")
    if "Asia/Tokyo" not in btpatch:
        fail("naive NPB datetimes are not explicitly interpreted as JST")

    for needle in (
        "npb_runtime_patch.py",
        "baseball_backtest_runtime_patch.py",
        "baseball_production_runtime_patch.py",
        "baseball_backtest.py",
        "source_quality_gate.py",
        "MLB_ENRICH_STARTERS: \"1\"",
    ):
        if needle not in workflow:
            fail(f"production workflow missing required reference: {needle}")
    if not re.search(r"NPB_MIN_STARTER_LINE_COVERAGE\s*:\s*['\"]?70(?:\.0)?['\"]?", workflow):
        fail("production workflow NPB starter coverage threshold missing")
    if not re.search(r"MLB_MIN_STARTER_COVERAGE\s*:\s*['\"]?90(?:\.0)?['\"]?", workflow):
        fail("production workflow MLB starter coverage threshold missing")
    if "if: always()" not in workflow or "actions/download-artifact@v4" not in workflow:
        fail("production workflow artifact recovery is not fail-safe")

    try:
        npb_policy = policy["NPB"]
        mlb_policy = policy["MLB"]
    except KeyError as e:
        fail(f"DATA_SOURCE_POLICY missing league section: {e}")
    if "npb.jp" not in json.dumps(npb_policy, ensure_ascii=False).lower():
        fail("NPB policy does not name official NPB as an authoritative source")
    if "statsapi.mlb.com" not in json.dumps(mlb_policy, ensure_ascii=False).lower():
        fail("MLB policy does not name MLB Stats API")

    for name, text in (("baseball_backtest.py", backtest), ("npb_multi_source.py", collector),
                       ("npb_runtime_patch.py", npbpatch), ("npb_official_schedule_patch.py", officialpatch),
                       ("baseball_backtest_runtime_patch.py", btpatch),
                       ("baseball_production_runtime_patch.py", prodpatch)):
        try:
            ast.parse(text, filename=name)
        except SyntaxError as e:
            fail(f"syntax error in {name}: {e}")

    for name in ("_update_pitcher_history", "match_features", "aggregate_npb_games"):
        assert_ast_function(backtest, "baseball_backtest.py", name)

    feature_pos = backtest.find("match_features(row)")
    update_pos = backtest.find("self._update_pitcher_history(row)")
    if feature_pos < 0 or update_pos < 0:
        fail("prediction/history call path missing")
    if feature_pos > update_pos:
        fail("pitcher history is updated before target feature generation")

    agg = DATA / "npb_multi_source_games_all.csv"
    if not agg.exists():
        print("[AUDIT] code/leakage-order/workflow/source-policy checks passed; aggregate data not present yet")
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
        print("[AUDIT WARN] aggregate is not globally sorted; loader must sort before walk-forward evaluation")

    if "date" in d.columns:
        date_dt = pd.to_datetime(d["date"], errors="coerce", utc=True)
        if date_dt.isna().any():
            fail("aggregate contains invalid date values")
        delta = (dt - date_dt).abs()
        if (delta > pd.Timedelta(days=1)).any():
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

    if "home_starter" in d.columns and "away_starter" in d.columns:
        both = (d["home_starter"].fillna("").astype(str).str.strip() != "") & (d["away_starter"].fillna("").astype(str).str.strip() != "")
        coverage = 100.0 * float(both.mean()) if len(d) else 0.0
        print(f"[AUDIT] both_starter_rows={int(both.sum())}/{len(d)} coverage_pct={coverage:.1f}")

    print("[AUDIT PASS] static, leakage-order, workflow, source-policy, and data-integrity checks passed")


if __name__ == "__main__":
    main()
