#!/usr/bin/env python3
"""Build a PIT-safe forward Matchday effect dataset from the live ledger.

Only settled predictions are included. Context events are derived from the
snapshot recorded at the original prediction timestamp; no postgame observation
is substituted. This dataset is then consumed by matchday_effect_fit.py.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from research.matchday_intelligence import Observation, is_pit_safe, usable_observations

RESULTS = Path("results")
BASELINE = RESULTS / "matchday_baseline.csv"
SNAPSHOTS = RESULTS / "matchday_snapshots.jsonl"
CHANGES = RESULTS / "matchday_change_ledger.jsonl"
OUTPUT = RESULTS / "matchday_effect_replay.csv"


def _load_snapshot_map() -> Dict[tuple[str, str], dict]:
    out: Dict[tuple[str, str], dict] = {}
    if not SNAPSHOTS.exists():
        return out
    for line in SNAPSHOTS.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            env = json.loads(line)
            key = (str(env["game_id"]), str(env["prediction_time_utc"]))
            out[key] = env
        except Exception:
            continue
    return out


def _safe_observations(env: dict, prediction_time: str):
    rows = []
    for item in env.get("observations", []):
        try:
            obs = Observation(**item)
            if is_pit_safe(obs):
                rows.append(obs)
        except Exception:
            continue
    return usable_observations(rows, prediction_time)


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    if not BASELINE.exists() or not SNAPSHOTS.exists():
        pd.DataFrame().to_csv(OUTPUT, index=False)
        print(json.dumps({"status":"DEFERRED","rows":0,"reason":"forward baseline/snapshot ledger missing"}, ensure_ascii=False))
        return 0

    df = pd.read_csv(BASELINE)
    if df.empty:
        pd.DataFrame().to_csv(OUTPUT, index=False)
        print(json.dumps({"status":"DEFERRED","rows":0,"reason":"forward baseline ledger empty"}, ensure_ascii=False))
        return 0

    df["actual"] = pd.to_numeric(df.get("actual"), errors="coerce")
    df = df[df["actual"].notna()].copy()
    # Effect learning must not treat repeated pregame snapshots of the same game
    # as independent outcomes. Keep only the final pregame forecast per game.
    if not df.empty:
        df["_pt_sort"] = pd.to_datetime(df["prediction_time_utc"], errors="coerce", utc=True)
        df = (
            df.sort_values(["game_id", "_pt_sort"], kind="mergesort")
              .drop_duplicates("game_id", keep="last")
              .drop(columns=["_pt_sort"])
              .reset_index(drop=True)
        )
    if df.empty:
        pd.DataFrame().to_csv(OUTPUT, index=False)
        print(json.dumps({"status":"DEFERRED","rows":0,"reason":"no settled forward predictions"}, ensure_ascii=False))
        return 0

    # Use one canonical final-pregame snapshot per game for causal/effect
    # learning; repeated interim snapshots remain available to change-ledger research.
    canonical_snapshot = {}
    all_snapshots_by_game = {}
    for (gid, pt), env in snapshots.items():
        current_pt = pd.to_datetime(pt, errors="coerce", utc=True)
        if pd.isna(current_pt):
            continue
        all_snapshots_by_game.setdefault(gid, []).append((current_pt, env))
        previous = canonical_snapshot.get(gid)
        if previous is None or current_pt > previous[0]:
            canonical_snapshot[gid] = (current_pt, env)
    snapshots = {(gid, str(env["prediction_time_utc"])): env for gid, (_pt, env) in canonical_snapshot.items()}

    changed_events = {}
    if CHANGES.exists():
        for line in CHANGES.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                ev = json.loads(line)
                key = (str(ev["game_id"]), str(ev["prediction_time_utc"]))
                changed_events.setdefault(key, set()).add(str(ev["event"]))
            except Exception:
                continue
    # Keep durable-change events, but model state transitions explicitly:
    # OUT is cleared by RETURNED; PROJECTED is cleared by CONFIRMED. This avoids
    # impossible final snapshots carrying both states at once.
    cumulative_events_by_game = {}
    event_values = {
        "STARTER_CONFIRMED","STARTER_CHANGED",
        "LINEUP_CONFIRMED","LINEUP_PROJECTED",
        "PLAYER_OUT","PLAYER_RETURNED","WEATHER_CHANGED",
        "REST_ASYMMETRY","TRAVEL_BURDEN","MARKET_MOVED",
        "BULLPEN_STATE_CHANGED",
    }
    transition_pairs = {
        "PLAYER_RETURNED": "PLAYER_OUT",
        "LINEUP_CONFIRMED": "LINEUP_PROJECTED",
    }
    for gid, seq in all_snapshots_by_game.items():
        cumulative = set()
        for _pt, env in sorted(seq, key=lambda x: x[0]):
            pt = str(env.get("prediction_time_utc"))
            events_now = set(changed_events.get((gid, pt), set()))
            for item in env.get("observations", []):
                value = item.get("value")
                if isinstance(value, str) and value in event_values:
                    events_now.add(value)
            for event_name in events_now:
                prior = transition_pairs.get(event_name)
                if prior:
                    cumulative.discard(prior)
                cumulative.add(event_name)
        cumulative_events_by_game[gid] = cumulative

    rows = []
    for rec in df.to_dict("records"):
        gid = str(rec.get("game_id") or "")
        pt = str(rec.get("prediction_time_utc") or "")
        env = snapshots.get((gid, pt))
        if env is None:
            continue
        obs = _safe_observations(env, pt)
        events = set(cumulative_events_by_game.get(gid, set()))
        for o in obs:
            value = str(o.value) if not isinstance(o.value, (dict, list)) else ""
            if value in {
                "STARTER_CONFIRMED",
                "STARTER_CHANGED",
                "LINEUP_CONFIRMED",
                "LINEUP_PROJECTED",
                "PLAYER_OUT",
                "PLAYER_RETURNED",
                "WEATHER_CHANGED",
                "REST_ASYMMETRY",
                "TRAVEL_BURDEN",
                "MARKET_MOVED",
                "BULLPEN_STATE_CHANGED",
            }:
                events.add(value)
            kind = str(o.kind).upper()
            if kind == "WEATHER" and o.state in {"VERIFIED","PROJECTED"}:
                events.add("WEATHER_PRESENT")
            if kind == "REST_TRAVEL" and o.state == "VERIFIED":
                events.add("REST_TRAVEL_PRESENT")

        for event in changed_events.get((gid, pt), set()):
            events.add(event)

        out = {
            "game_id": gid,
            "datetime": rec.get("datetime"),
            "prediction_time_utc": pt,
            "actual": int(rec["actual"]),
            "pred_context_free_home": float(rec["pred_home"]),
            "pred_context_free_draw": float(rec.get("pred_draw", float("nan"))),
            "pred_context_free_away": float(rec["pred_away"]),
            "baseline_context_free": True,
        }
        for event in sorted(events):
            out[f"ctx_{event}"] = 1
        rows.append(out)

    out_df = pd.DataFrame(rows)
    if not out_df.empty:
        out_df.to_csv(OUTPUT, index=False)
    else:
        pd.DataFrame(columns=["game_id","datetime","prediction_time_utc","actual"]).to_csv(OUTPUT,index=False)

    payload = {
        "status": "PASS" if len(out_df) else "DEFERRED",
        "rows": int(len(out_df)),
        "settled_baseline_rows": int(len(df)),
        "snapshot_matched_rows": int(len(out_df)),
        "event_columns": sorted([c for c in out_df.columns if c.startswith("ctx_")]),
        "pit_rule": "only snapshot observations with available_at <= prediction_time are retained",
    }
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    import json
    raise SystemExit(main())
