#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Autonomous baseball research controller.

This controller never changes a model because of a single noisy result. It
reads the latest OOS reports, ranks weaknesses, compares available model
candidates, updates a persistent research state, and emits the next research
objective. All decisions are based on files produced by the chronological
backtest; no target-game information is introduced here.
"""
from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
STATE = ROOT / "research_state.json"
HISTORY = ROOT / "research_history.json"
PLAN = ROOT / "research_plan.json"

OBJECTIVES = [
    ("win", "勝敗確率の校正とモデル重み", ["Accuracy", "LogLoss", "Brier"]),
    ("score", "得点分布・スコア候補", ["MeanAbsoluteScoreError", "Top4ScoreHitRate"]),
    ("low_high", "Low/High境界と確率校正", ["LowHighAccuracy"]),
    ("starter", "先発投手情報の品質と寄与", ["starter_coverage", "starter_accuracy"]),
    ("bullpen", "ブルペン疲労・救援力", ["HighAccuracy", "MeanAbsoluteScoreError"]),
    ("batting", "打線・対左右・直近フォーム", ["Accuracy", "MeanAbsoluteScoreError"]),
    ("environment", "球場・天候・日程環境", ["MeanAbsoluteScoreError", "LowHighAccuracy"]),
]


def read_csv(name: str) -> pd.DataFrame:
    p = RESULTS / name
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()


def finite(x: Any, default=float("nan")) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def metric_direction(metric: str) -> int:
    return -1 if metric in {"LogLoss", "Brier", "MeanAbsoluteScoreError", "RMSE"} else 1


def latest_summary(league: str) -> Dict[str, Any]:
    df = read_csv(f"{league.lower()}_backtest_summary.csv")
    return {} if df.empty else df.iloc[-1].to_dict()


def model_leaderboard(league: str) -> List[Dict[str, Any]]:
    df = read_csv(f"{league.lower()}_model_comparison.csv")
    if df.empty or "model" not in df.columns:
        return []
    rows = df.to_dict("records")
    # Multi-objective rank: prioritize log loss, then Brier, then accuracy.
    def key(r):
        ll = finite(r.get("LogLoss"), 99.0)
        br = finite(r.get("Brier"), 99.0)
        ac = finite(r.get("Accuracy"), 0.0)
        return (ll, br, -ac)
    return sorted(rows, key=key)


def detailed_weakness(league: str) -> Dict[str, Any]:
    df = read_csv(f"{league.lower()}_accuracy_detail.csv")
    if df.empty:
        return {}
    out: Dict[str, Any] = {}
    for col in ("home_team", "away_team", "model", "condition", "segment"):
        if col in df.columns:
            out[col] = int(df[col].nunique(dropna=True))
    for metric in ("Accuracy", "LogLoss", "Brier", "MeanAbsoluteScoreError", "LowHighAccuracy", "HighAccuracy", "LowAccuracy"):
        if metric in df.columns:
            out[metric] = finite(df[metric].mean())
    return out


def build_state() -> Dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    leagues = {}
    weaknesses = []
    for league in ("NPB", "MLB"):
        summary = latest_summary(league)
        leaderboard = model_leaderboard(league)
        detail = detailed_weakness(league)
        leagues[league] = {
            "summary": summary,
            "model_leaderboard": leaderboard,
            "detail": detail,
        }
        if summary:
            # Convert observed weakness into a comparable priority score.
            acc = finite(summary.get("Accuracy"), 0.5)
            ll = finite(summary.get("LogLoss"), 1.0)
            brier = finite(summary.get("Brier"), 0.25)
            score_mae = finite(summary.get("MeanAbsoluteScoreError"), 3.0)
            lh = finite(summary.get("LowHighAccuracy"), 0.5)
            weaknesses.extend([
                {"league": league, "objective": "win", "priority": max(0.0, 0.65-acc) + max(0.0, ll-0.67), "reason": "勝敗精度/確率損失"},
                {"league": league, "objective": "score", "priority": max(0.0, score_mae-2.0) * 0.35, "reason": "スコア誤差"},
                {"league": league, "objective": "low_high", "priority": max(0.0, 0.68-lh), "reason": "Low/High精度"},
                {"league": league, "objective": "calibration", "priority": max(0.0, brier-0.25), "reason": "確率校正"},
            ])
    weaknesses.sort(key=lambda x: x["priority"], reverse=True)
    focus = weaknesses[0] if weaknesses else {"league": "NPB", "objective": "win", "priority": 1.0, "reason": "評価データ待ち"}
    return {
        "schema_version": 1,
        "updated_at": now,
        "goal": "maximize validated out-of-sample accuracy for every prediction item",
        "processing_budget_seconds": int(os.getenv("BASEBALL_TIME_BUDGET_SEC", "3600")),
        "focus": focus,
        "weaknesses": weaknesses[:20],
        "leagues": leagues,
        "next_research": {
            "objective": focus["objective"],
            "league": focus["league"],
            "reason": focus["reason"],
            "rule": "candidate must improve the target metric OOS without unacceptable regression in other tracked metrics",
        },
    }


def append_history(state: Dict[str, Any]) -> None:
    history: List[Dict[str, Any]] = []
    if HISTORY.exists():
        try:
            raw = json.loads(HISTORY.read_text(encoding="utf-8"))
            if isinstance(raw, list): history = raw
        except Exception:
            pass
    history.append({
        "timestamp": state["updated_at"],
        "focus": state["focus"],
        "next_research": state["next_research"],
        "model_winners": {
            league: (data["model_leaderboard"][0] if data["model_leaderboard"] else None)
            for league, data in state["leagues"].items()
        },
    })
    HISTORY.write_text(json.dumps(history[-500:], ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    state = build_state()
    STATE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    append_history(state)
    PLAN.write_text(json.dumps({
        "generated_at": state["updated_at"],
        "priority": state["focus"],
        "next": state["next_research"],
        "automatic_adoption_policy": {
            "require_oos": True,
            "require_no_material_regression": True,
            "keep_failed_candidates_for_learning": True,
            "never_use_future_target_data": True,
        },
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"focus": state["focus"], "next": state["next_research"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
