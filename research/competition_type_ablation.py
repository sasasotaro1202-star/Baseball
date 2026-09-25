#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Two-policy OOS ablation for a single competition category.

The runner compares the repository default training policy against the same
policy plus one selected competition category. Evaluation categories remain
unchanged, and each policy gets its own checkpoint namespace in the backtest
engine. No production promotion is performed.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SNAP = RESULTS / "competition_type_ablation"


BASELINE = {
    "NPB": ("regular", "interleague"),
    "MLB": ("regular",),
    "INTERNATIONAL": (),
}

EVALUATION = {
    "NPB": ("regular", "interleague", "climax", "japan_series", "allstar", "special"),
    "MLB": (
        "regular", "wild_card", "division_series",
        "league_championship_series", "world_series",
        "postseason", "allstar", "championship", "exhibition",
    ),
    "INTERNATIONAL": (
        "wbc", "premier12", "olympics", "asian_games",
        "asian_professional_baseball_championship",
        "international_friendly", "international_special",
        "other_international",
    ),
}


def run_variant(league: str, categories: tuple[str, ...], out_dir: Path, data_dir: Path) -> dict:
    env = os.environ.copy()
    env["BASEBALL_TIME_BUDGET_SEC"] = env.get("BASEBALL_TIME_BUDGET_SEC", "1500")
    env[f"BASEBALL_{'INTL' if league == 'INTERNATIONAL' else league}_TRAINING_GAME_CATEGORIES"] = ",".join(categories)
    env[f"BASEBALL_{'INTL' if league == 'INTERNATIONAL' else league}_EVALUATION_GAME_CATEGORIES"] = ",".join(EVALUATION[league])

    flag = {
        "NPB": "--npb-only",
        "MLB": "--mlb-only",
        "INTERNATIONAL": "--international-only",
    }[league]
    cmd = [sys.executable, str(ROOT / "baseball_backtest.py"), flag, "--data-dir", str(data_dir)]
    log = out_dir / "stdout_stderr.log"
    out_dir.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as fh:
        proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT, check=False)

    copied = []
    for name in (
        f"{league.lower()}_backtest_summary.csv",
        f"{league.lower()}_competition_type_metrics.csv",
        f"{league.lower()}_model_comparison.csv",
    ):
        src = RESULTS / name
        if src.exists() and src.stat().st_size > 0:
            dst = out_dir / name
            shutil.copy2(src, dst)
            copied.append(str(dst))

    return {
        "status": "PASS" if proc.returncode == 0 else "FAIL",
        "returncode": proc.returncode,
        "training_categories": list(categories),
        "evaluation_categories": list(EVALUATION[league]),
        "artifacts": copied,
        "log": str(log),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--league", choices=("NPB", "MLB", "INTERNATIONAL"), required=True)
    ap.add_argument("--category", required=True)
    ap.add_argument("--data-dir", default="data")
    args = ap.parse_args()

    league = args.league
    category = args.category.strip().lower()
    if category not in EVALUATION[league]:
        raise SystemExit(f"category is not in the league evaluation universe: {category}")

    baseline = tuple(BASELINE[league])
    candidate = tuple(dict.fromkeys((*baseline, category)))

    # There is no meaningful include/exclude comparison when the selected
    # category is already in the baseline training policy.
    if candidate == baseline:
        payload = {
            "schema_version": 1,
            "status": "DEFERRED",
            "reason": "selected category is already in baseline training policy",
            "league": league,
            "category": category,
        }
        SNAP.mkdir(parents=True, exist_ok=True)
        (SNAP / f"{league.lower()}_{category}_comparison.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(payload, ensure_ascii=False))
        return 0

    root = SNAP / league.lower() / category
    baseline_result = run_variant(league, baseline, root / "baseline", Path(args.data_dir))
    candidate_result = run_variant(league, candidate, root / "candidate", Path(args.data_dir))

    summary = {
        "schema_version": 1,
        "league": league,
        "category": category,
        "status": "PASS",
        "promotion_auto": False,
        "baseline": baseline_result,
        "candidate": candidate_result,
        "adoption_rule": "Require same chronological OOS population, PIT integrity, calibration, robustness, latest untouched data and frozen-holdout gates; no automatic promotion.",
    }
    (root / "comparison.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
