#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deterministic integrity, leakage, workflow, and data-quality audit."""
from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

REQUIRED = (
    ".github/workflows/baseball_production.yml",
    ".github/workflows/baseball_audit.yml",
    "baseball_backtest.py",
    "baseball_backtest_runtime_patch.py",
    "baseball_production_runtime_patch.py",
    "baseball_mlb_score_hilo_patch.py",
    "baseball_quality_runtime_patch.py",
    "npb_multi_source.py",
    "npb_runtime_patch.py",
    "npb_official_schedule_patch.py",
    "npb_quality_runtime_patch.py",
    "source_quality_gate.py",
    "repair_npb_syntax.py",
    "DATA_SOURCE_POLICY.json",
)


def fail(msg: str) -> None:
    raise SystemExit("[AUDIT FAIL] " + msg)


def require(text: str, needle: str, label: str) -> None:
    if needle not in text:
        fail(f"{label}: missing {needle}")


def assert_ast_function(text: str, filename: str, name: str) -> None:
    try:
        tree = ast.parse(text, filename=filename)
    except SyntaxError as e:
        fail(f"syntax error in {filename}: {e}")
    if not any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name for n in ast.walk(tree)):
        fail(f"{filename}: function {name} missing")


def main() -> None:
    repair = ROOT / "repair_npb_syntax.py"
    if repair.exists():
        r = subprocess.run([sys.executable, str(repair)], cwd=ROOT, text=True, capture_output=True)
        if r.returncode != 0:
            fail("deterministic source repair failed: " + (r.stderr.strip() or r.stdout.strip()))

    for rel in REQUIRED:
        if not (ROOT / rel).exists():
            fail(f"required file missing: {rel}")

    files = {rel: (ROOT / rel).read_text(encoding="utf-8") for rel in REQUIRED if not rel.endswith(".json")}
    policy = json.loads((ROOT / "DATA_SOURCE_POLICY.json").read_text(encoding="utf-8"))
    workflow = files[".github/workflows/baseball_production.yml"]
    backtest = files["baseball_backtest.py"]
    score_patch = files["baseball_backtest_runtime_patch.py"]
    prod_patch = files["baseball_production_runtime_patch.py"]
    mlb_patch = files["baseball_mlb_score_hilo_patch.py"]
    quality_patch = files["baseball_quality_runtime_patch.py"]
    npb_patch = files["npb_runtime_patch.py"]
    official_patch = files["npb_official_schedule_patch.py"]
    npb_quality = files["npb_quality_runtime_patch.py"]

    # Production workflow contract.
    for needle in (
        "concurrency:",
        "group: baseball-production-v8",
        "cancel-in-progress: false",
        "runs-on: [self-hosted, macOS, X64]",
        "baseball_backtest_runtime_patch.py",
        "baseball_production_runtime_patch.py",
        "npb_official_schedule_patch.py",
        "npb_quality_runtime_patch.py",
        "baseball_mlb_score_hilo_patch.py",
        "baseball_quality_runtime_patch.py",
        "NPB_COLLECTION_BUDGET_SEC: \"3600\"",
        "BASEBALL_TIME_BUDGET_SEC: \"3600\"",
        "MLB_MIN_STARTER_COVERAGE: \"90\"",
        "timeout-minutes: 140",
        "requirements.txt",
        "REQ_HASH=",
        "requirements unchanged; reusing persistent environment",
        "return 1",
    ):
        require(workflow, needle, "production workflow")

    # The runtime patch chain must be ordered before the production patch that
    # adds starter/data hardening, and every patch must fail closed.
    order = [
        "apply_patch_safely npb_runtime_patch.py",
        "apply_patch_safely npb_official_schedule_patch.py",
        "apply_patch_safely npb_quality_runtime_patch.py",
        "apply_patch_safely baseball_backtest_runtime_patch.py",
        "apply_patch_safely baseball_production_runtime_patch.py",
        "apply_patch_safely baseball_mlb_score_hilo_patch.py",
        "apply_patch_safely baseball_quality_runtime_patch.py",
    ]
    pos = -1
    for needle in order:
        nxt = workflow.find(needle)
        if nxt <= pos:
            fail(f"patch ordering invalid at {needle}")
        pos = nxt

    for marker, text in (
        ("BACKTEST_RUNTIME_HARDENING_V4", score_patch),
        ("BASEBALL_PRODUCTION_HARDENING_V2", prod_patch),
        ("MLB_SCORE_HILO_PATCH_V2", mlb_patch),
        ("BASEBALL_QUALITY_HARDENING_V1", quality_patch),
        ("RUNTIME_HARDENING_V5", npb_patch),
        ("OFFICIAL_NPB_SCHEDULE_VALIDATION_V1", official_patch),
        ("NPB_QUALITY_HARDENING_V1", npb_quality),
    ):
        require(text, marker, "hardening marker")

    # Adaptive score layer is actually applied in production, not merely
    # present in the repository.
    for needle in ("_score_prior", "prior_blend", "_nb_nll", "dispersion_home", "dispersion_away"):
        require(score_patch, needle, "adaptive score layer")
    require(workflow, "baseball_backtest_runtime_patch.py", "adaptive score workflow application")

    # Starter/data quality contracts.
    require(mlb_patch, "confirmed_starters", "MLB starter gate")
    require(mlb_patch, "starter_rate < 0.90", "MLB starter gate")
    require(mlb_patch, "weather_forecast_asof_cutoff", "MLB weather leakage guard")
    require(npb_patch, "_official_starters_from_npb", "NPB official starter resolver")
    require(npb_patch, "Strict rule: unresolved starters remain unresolved", "NPB starter strictness")
    require(npb_quality, "EMPTY SCHEDULE -> preserved checkpoint", "NPB empty schedule protection")
    require(npb_quality, "repair_or_missing", "NPB stale checkpoint repair")
    require(official_patch, "_official_validate_games", "official NPB schedule validation")
    require(official_patch, "schedule_{month:02d}_detail.html", "official NPB schedule source")
    require(score_patch, "Asia/Tokyo", "NPB datetime interpretation")

    # Source files parse cleanly.
    for rel, text in files.items():
        try:
            ast.parse(text, filename=rel)
        except SyntaxError as e:
            fail(f"syntax error in {rel}: {e}")

    for name in ("_update_pitcher_history", "match_features", "aggregate_npb_games", "fit_score_ensemble", "predict_scores"):
        assert_ast_function(backtest, "baseball_backtest.py", name)

    # Explicit leakage-order invariant: target features are generated before
    # pitcher history is updated with the target game.
    feature_pos = backtest.find("match_features(row)")
    update_pos = backtest.find("self._update_pitcher_history(row)")
    if feature_pos < 0 or update_pos < 0 or feature_pos > update_pos:
        fail("prediction/history ordering invariant violated")

    # Source-policy sanity.
    if "npb.jp" not in json.dumps(policy.get("NPB", {}), ensure_ascii=False).lower():
        fail("NPB policy does not identify official NPB")
    if "statsapi.mlb.com" not in json.dumps(policy.get("MLB", {}), ensure_ascii=False).lower():
        fail("MLB policy does not identify MLB Stats API")

    # Data checks are conditional because a clean checkout before the first
    # successful collector run legitimately has no aggregate yet.
    agg = DATA / "npb_multi_source_games_all.csv"
    if not agg.exists():
        print("[AUDIT] code/leakage/workflow/source-policy checks passed; aggregate data not present yet")
        print("[AUDIT PASS]")
        return

    try:
        d = pd.read_csv(agg, low_memory=False)
    except Exception as e:
        fail(f"cannot read aggregate: {e}")
    if "game_id" not in d.columns:
        fail("aggregate has no game_id")
    ids = d.game_id.astype(str).str.replace(r"\.0$", "", regex=True)
    dup = int(ids.duplicated().sum())
    if dup:
        fail(f"aggregate contains {dup} duplicate game IDs")
    if "datetime" not in d.columns:
        fail("aggregate has no datetime")
    dt = pd.to_datetime(d.datetime, errors="coerce", utc=True)
    if dt.isna().any():
        fail(f"aggregate has {int(dt.isna().sum())} invalid datetimes")

    print(f"[AUDIT] aggregate games={len(d)} unique_game_ids={ids.nunique()}")
    print("[AUDIT PASS] static, leakage-order, workflow, source-policy, and data-integrity checks passed")


if __name__ == "__main__":
    main()
