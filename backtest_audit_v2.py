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

matchday_runner = need("production_matchday_intelligence.py", r"def\\s+main\\b", "current Matchday Intelligence runner")
if matchday_runner:
    for token, label in [
        ("fetch_official_starters(target_date)", "target-date starter lookup"),
        ("fetch_lineup(g["game_id"],target_date", "date-aware lineup lookup"),
        ("shadow_not_promoted", "shadow non-promotion marker"),
        ("matchday_policy_eligible", "Matchday Policy eligibility marker"),
    ]:
        if token not in matchday_runner:
            errors.append(f"missing {label} in production_matchday_intelligence.py")

matchday_replay = need("research/matchday_replay.py", r"class|def\\s+main\\b", "Matchday PIT replay")
if matchday_replay:
    for token, label in [
        ("usable_observations", "PIT observation filter"),
        ("matchday_replay.json", "replay artifact"),
        ("DEFERRED", "fail-closed replay state"),
    ]:
        if token not in matchday_replay:
            errors.append(f"missing {label} in research/matchday_replay.py")

weather_ingest = need("npb_multi_source.py", r"historical-forecast-api\.open-meteo\.com", "PIT-safe historical forecast weather source")
if weather_ingest:
    for token, label in [
        ("weather_available_at", "weather availability timestamp"),
        ("weather_pit_quality", "weather PIT quality marker"),
        ("CONSERVATIVE_8H_BOUND", "conservative weather availability bound"),
    ]:
        if token not in weather_ingest:
            errors.append(f"missing {label} in npb_multi_source.py")

matchday_workflow = need(".github/workflows/baseball_matchday.yml", r"baseball-matchday", "Matchday Intelligence workflow")
if matchday_workflow:
    for token, label in [
        ('cron: "*/30 * * * *"', "30-minute matchday schedule"),
        ("production_matchday_intelligence.py", "live matchday runner"),
        ("shadow_not_promoted", "non-promotion safety marker"),
        ("runs-on: ubuntu-latest", "standard hosted runner"),
        ("contents: write", "matchday state persistence permission"),
    ]:
        if token not in matchday_workflow:
            errors.append(f"missing {label} in baseball_matchday.yml")

conformal = need("research/conformal_uncertainty.py", r"def\s+uncertainty_summary\b", "conformal uncertainty")
if conformal:
    for token, label in [
        ("def class_pvalues", "conformal class p-values"),
        ("def prediction_set", "conformal prediction set"),
        ("does not alter probabilities", "probability-preserving uncertainty policy"),
    ]:
        if token not in conformal:
            errors.append(f"missing {label} in research/conformal_uncertainty.py")

routing = need("research/drift_uncertainty_routing.py", r"def\s+route_experts\b", "drift/uncertainty routing")
if routing:
    for token, label in [
        ("def mmd_drift_score", "MMD drift signal"),
        ("class AdaptiveTemperatureCalibrator", "adaptive recalibration"),
        ("uncertainty_disagreement_mix", "disagreement-aware uncertainty"),
    ]:
        if token not in routing:
            errors.append(f"missing {label} in research/drift_uncertainty_routing.py")

matchday = need("research/matchday_intelligence.py", r"class\s+ContextState", "Matchday Intelligence PIT layer")
if matchday:
    for token, label in [
        ("def is_pit_safe", "matchday PIT check"),
        ("def usable_observations", "available-at filter"),
        ("ContextEvent.STARTER_CHANGED", "context change ledger"),
    ]:
        if token not in matchday:
            errors.append(f"missing {label} in research/matchday_intelligence.py")

effect_fit = need("research/matchday_effect_fit.py", r"def\s+fit_effects\b", "Matchday effect learner")
if effect_fit:
    for token, label in [
        ("baseline_context_free", "context-free baseline guard"),
        ("validation_delta", "Matchday OOS validation"),
        ("artifact_type", "Matchday effect artifact schema"),
    ]:
        if token not in effect_fit:
            errors.append(f"missing {label} in research/matchday_effect_fit.py")

reforecast = need("research/matchday_reforecast.py", r"def\s+reforecast\b", "Matchday reforecast controller")
if reforecast:
    for token, label in [
        ("FileNotFoundError", "fail-closed effect loading"),
        ("fusion_alpha", "learned Matchday fusion weight"),
        ("NO_LEARNED_EFFECT", "unknown effect skip"),
    ]:
        if token not in reforecast:
            errors.append(f"missing {label} in research/matchday_reforecast.py")

routing_gate = need("research/routing_acceptance_gate.py", r"MIN_LOGLOSS_IMPROVEMENT", "routing acceptance gate")
if routing_gate:
    for token, label in [
        ("MIN_BRIER_IMPROVEMENT", "routing Brier gate"),
        ("MAX_ECE_REGRESSION", "routing ECE gate"),
        ("requires_two_non_overlapping_late_oos_windows", "two-window OOS gate"),
    ]:
        if token not in routing_gate:
            errors.append(f"missing {label} in research/routing_acceptance_gate.py")

frozen = need("research/frozen_holdout_gate.py", r"HOLDOUT_FRAC|MIN_HOLDOUT_ROWS", "frozen holdout gate")
if frozen:
    for token in ("datetime", "tuning_rule", "status"):
        if token not in frozen:
            errors.append(f"frozen holdout gate missing {token}")

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
