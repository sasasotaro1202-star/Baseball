#!/usr/bin/env python3
"""Deterministic preflight audit for the baseball backtest pipeline."""
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parent
errors = []


def need(path, pattern, label):
    p = ROOT / path
    if not p.exists():
        errors.append(f"missing {path}")
        return ""
    text = p.read_text(encoding="utf-8", errors="replace")
    if not re.search(pattern, text, re.S):
        errors.append(f"missing {label} in {path}")
    return text

npbpatch = need("npb_runtime_patch.py", r"# RUNTIME_HARDENING_V5", "NPB runtime hardening V5")
if npbpatch:
    for token, label in [
        ("_official_starters_from_npb", "strict official starter resolver"),
        ("def first_pitchers(", "starter resolver"),
        ("first_pitchers(r.game_id,r.home,r.away)", "team-aware official fallback call"),
        ("starter_source", "starter provenance"),
        ("weather_source", "weather provenance"),
    ]:
        if token not in npbpatch:
            errors.append(f"missing {label} in npb_runtime_patch.py")

runtime = need("baseball_backtest_runtime_patch.py", r"# BACKTEST_RUNTIME_HARDENING_V4", "backtest runtime hardening V4")
if runtime:
    for token, label in [
        ("def _score_prior", "structural run prior"),
        ("def fit_score_ensemble", "chronological score ensemble"),
        ("prior_blend", "prior/model calibration"),
        ("dispersion_home", "overdispersion"),
    ]:
        if token not in runtime:
            errors.append(f"missing {label} in baseball_backtest_runtime_patch.py")

base = need("baseball_backtest.py", r"def\s+_update_pitcher_history\b", "pitcher history updater")
if base:
    pred = base.find("match_features(row)")
    hist = base.find("_update_pitcher_history(row)")
    if pred >= 0 and hist >= 0 and hist < pred:
        errors.append("pitcher history update occurs before prediction features")

workflow = need(".github/workflows/baseball_production.yml", r"timeout-minutes:\s*140", "production job limit")
if workflow:
    for token in ("BASEBALL_TIME_BUDGET_SEC", "data/checkpoints/npb_collection_status.json", "contents: write"):
        if token not in workflow:
            errors.append(f"workflow invariant missing: {token}")
audit_workflow = need(".github/workflows/baseball_audit.yml", r"backtest_audit_v2\.py", "audit workflow")
if audit_workflow and "backtest_audit.py" in audit_workflow:
    errors.append("stale audit workflow still references removed backtest_audit.py")

validation = need(".github/workflows/validate-code.yml", r"\.github/workflows/\*\*", "workflow-wide validation trigger")
if validation and "[YAML] PASS" not in validation:
    errors.append("workflow-wide YAML validation step missing")

production = need(".github/workflows/baseball_production.yml", r"Validate Code Syntax", "production validation dependency")
if production and ("workflow_run:" not in production or "github.event.workflow_run.conclusion == 'success'" not in production):
    errors.append("production is not fail-closed on successful validation")

for workflow_path in (
    ".github/workflows/baseball_production.yml",
    ".github/workflows/baseball-data-acquisition.yml",
    ".github/workflows/baseball_mac_compute.yml",
    ".github/workflows/baseball_research.yml",
    ".github/workflows/baseball_recovery.yml",
):
    w = need(workflow_path, r"runs-on:\s+ubuntu-latest", "hosted Linux runner")
    if w and re.search(r"runs-on:\s*self-hosted", w):
        errors.append(f"self-hosted runner reference remains in {workflow_path}")

try:
    import pandas as pd
    candidates = [ROOT / "data" / "npb_multi_source_games_all.csv", ROOT / "data" / "npb_aggregate.csv"]
    existing = next((p for p in candidates if p.exists()), None)
    if existing is not None:
        df = pd.read_csv(existing)
        if "game_id" in df.columns and df["game_id"].duplicated().any():
            errors.append("duplicate game_id values in aggregate data")
        if "datetime" in df.columns:
            dt = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
            if dt.isna().all():
                errors.append("no valid timestamps in aggregate datetime")
except Exception as exc:
    errors.append(f"aggregate audit failed: {exc}")

if errors:
    print("[AUDIT] FAIL")
    for e in errors:
        print(f"- {e}")
    sys.exit(1)
print("[AUDIT] PASS: code and available-data invariants are valid")
