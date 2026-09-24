#!/usr/bin/env python3
"""Research-only PIT replay for Matchday Intelligence.

Inputs:
- results/matchday_baseline.csv with game_id, prediction_time_utc, actual,
  pred_home/pred_away (+ pred_draw for NPB).
- results/matchday_snapshots.jsonl where each line is a JSON snapshot envelope
  containing game_id, prediction_time_utc and an observations list compatible
  with research.matchday_intelligence.Observations.

Only observations provably available at the prediction timestamp are passed to
the Matchday Policy. Missing timing evidence therefore produces a baseline
prediction, not an inferred adjustment.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from research.matchday_intelligence import Observation, is_pit_safe, usable_observations
from research.matchday_policy import apply_matchday_policy, load_policy

RESULTS = Path("results")
BASELINE = RESULTS / "matchday_baseline.csv"
SNAPSHOTS = RESULTS / "matchday_snapshots.jsonl"
OUTPUT = RESULTS / "matchday_replay.json"


def metrics(p: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=int)
    p = np.clip(p, 1e-12, 1.0)
    p /= p.sum(axis=1, keepdims=True)
    target = np.zeros_like(p)
    target[np.arange(len(y)), y] = 1.0
    ll = float(np.mean(-np.log(p[np.arange(len(y)), y])))
    brier = float(np.mean(np.sum((p - target) ** 2, axis=1)))
    acc = float(np.mean(p.argmax(axis=1) == y))
    return {"logloss": ll, "brier": brier, "accuracy": acc}


def load_snapshots() -> Dict[str, List[Observation]]:
    out: Dict[str, List[Observation]] = {}
    if not SNAPSHOTS.exists():
        return out
    for line in SNAPSHOTS.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            env = json.loads(line)
            gid = str(env["game_id"])
            pt = str(env["prediction_time_utc"])
            raw = []
            for item in env.get("observations", []):
                raw.append(Observation(**item))
            out.setdefault(gid, []).append(raw)
            # Store the timestamp on the envelope separately; duplicate snapshots
            # are retained because an official lineup can change later.
            out[gid] = out[gid]
        except Exception:
            continue
    return out


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    if not BASELINE.exists() or not SNAPSHOTS.exists():
        payload = {
            "status": "DEFERRED",
            "eligible": False,
            "reason": "matchday baseline or snapshot ledger is missing",
        }
        OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(payload, ensure_ascii=False))
        return 0

    df = pd.read_csv(BASELINE)
    required = {"game_id", "prediction_time_utc", "actual", "pred_home", "pred_away"}
    if not required.issubset(df.columns) or df.empty:
        payload = {"status": "DEFERRED", "eligible": False, "reason": "baseline schema is incomplete"}
        OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(payload, ensure_ascii=False))
        return 0

    # Forward ledger rows are written before outcomes exist. Only settled rows
    # enter scoring; unresolved NaN/blank outcomes are preserved for later runs.
    df = df[pd.to_numeric(df["actual"], errors="coerce").notna()].copy()
    if df.empty:
        payload = {
            "status": "DEFERRED",
            "eligible": False,
            "reason": "No settled forward predictions are available yet.",
            "replayable_rows": 0,
        }
        OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(payload, ensure_ascii=False))
        return 0

    policy = load_policy()
    snapshot_rows = []
    for line in SNAPSHOTS.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            env = json.loads(line)
            snapshot_rows.append(env)
        except Exception:
            continue

    selected = []
    for _, row in df.iterrows():
        gid = str(row["game_id"])
        cutoff = str(row["prediction_time_utc"])
        choices = [x for x in snapshot_rows if str(x.get("game_id")) == gid and str(x.get("prediction_time_utc")) == cutoff]
        if not choices:
            selected.append(None)
            continue
        # Select the latest snapshot at the requested prediction timestamp.
        selected.append(choices[-1])

    base_probs = []
    final_probs = []
    outcomes = []
    used = 0
    for row, env in zip(df.to_dict("records"), selected):
        if env is None:
            continue
        pt = str(row["prediction_time_utc"])
        obs = []
        for item in env.get("observations", []):
            try:
                o = Observation(**item)
                if is_pit_safe(o):
                    obs.append(o)
            except Exception:
                continue
        context: Dict[str, Dict[str, Any]] = {}
        for o in usable_observations(obs, pt):
            context.setdefault(str(o.kind).lower(), {
                "state": o.state,
                "confidence": o.confidence if o.confidence is not None else 1.0,
                "snapshot_id": o.snapshot_id,
            })
        p = [float(row["pred_home"])]
        if "pred_draw" in row:
            p.append(float(row["pred_draw"]))
        p.append(float(row["pred_away"]))
        decision = apply_matchday_policy(np.asarray(p), context, policy=policy)
        base_probs.append(p)
        final_probs.append(decision.probabilities.tolist())
        outcomes.append(int(row["actual"]))
        used += 1

    if used < 50:
        payload = {
            "status": "DEFERRED",
            "eligible": False,
            "reason": "fewer than 50 replayable PIT snapshots",
            "replayable_rows": used,
        }
        OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(payload, ensure_ascii=False))
        return 0

    bm = metrics(np.asarray(base_probs), np.asarray(outcomes))
    fm = metrics(np.asarray(final_probs), np.asarray(outcomes))
    payload = {
        "schema_version": 1,
        "status": "PASS",
        "eligible": False,
        "replayable_rows": used,
        "baseline": bm,
        "matchday": fm,
        "delta": {f"delta_{k}": float(fm[k] - bm[k]) for k in bm},
        "policy_eligible": bool(policy.get("eligible", False)),
        "note": "This replay scores only observations with explicit availability evidence.",
    }
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
