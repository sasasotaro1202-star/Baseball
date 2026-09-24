#!/usr/bin/env python3
"""Build a chronological Matchday change/event ledger from saved PIT snapshots.

This is target-free: it compares only observations that were already available
at each prediction timestamp. It is useful for learning conditional context
effects such as starter changes, lineup confirmation, weather shifts and
availability changes.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from research.matchday_intelligence import Observation, is_pit_safe, usable_observations

RESULTS = Path("results")
INPUT = RESULTS / "matchday_snapshots.jsonl"
OUTPUT = RESULTS / "matchday_change_ledger.jsonl"


def _load() -> List[dict]:
    rows=[]
    if not INPUT.exists():
        return rows
    for line in INPUT.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            env=json.loads(line)
            env["game_id"]=str(env["game_id"])
            env["prediction_time_utc"]=str(env["prediction_time_utc"])
            rows.append(env)
        except Exception:
            continue
    return sorted(rows,key=lambda x:(x["game_id"],x["prediction_time_utc"]))


def _obs_map(env: dict) -> Dict[str, Observation]:
    result={}
    pt=str(env["prediction_time_utc"])
    for item in env.get("observations",[]):
        try:
            o=Observation(**item)
            if not is_pit_safe(o):
                continue
            usable=usable_observations([o],pt)
            for u in usable:
                result[str(u.kind).upper()+"::"+str(u.source)] = u
        except Exception:
            continue
    return result


def _num_weather(value: Any) -> List[float]:
    if not isinstance(value,dict):
        return []
    out=[]
    for key in ("temperature_c","humidity_pct","precip_mm","wind_kmh","wind_direction_deg"):
        v=value.get(key)
        try:
            out.append(float(v))
        except Exception:
            out.append(float("nan"))
    return out


def main() -> int:
    RESULTS.mkdir(parents=True,exist_ok=True)
    rows=_load()
    if not rows:
        OUTPUT.write_text("",encoding="utf-8")
        print(json.dumps({"status":"DEFERRED","events":0,"reason":"snapshot ledger is empty"},ensure_ascii=False))
        return 0

    by_game: Dict[str,List[dict]]={}
    for env in rows:
        by_game.setdefault(env["game_id"],[]).append(env)

    emitted=[]
    for gid, seq in by_game.items():
        previous=None
        for env in seq:
            current=_obs_map(env)
            events=[]
            if previous is not None:
                prevmap=previous

                for kind in ("STARTER","LINEUP","WEATHER","AVAILABILITY","REST_TRAVEL"):
                    cur_candidates=[(k,o) for k,o in current.items() if k.startswith(kind+"::")]
                    prev_candidates=[(k,o) for k,o in prevmap.items() if k.startswith(kind+"::")]
                    cur=cur_candidates[0][1] if cur_candidates else None
                    prev=prev_candidates[0][1] if prev_candidates else None
                    if cur is None:
                        continue

                    if kind=="STARTER":
                        if prev is not None and prev.value != cur.value:
                            events.append("STARTER_CHANGED")
                        if str(cur.state).upper()=="VERIFIED" and (
                            prev is None or str(prev.state).upper()!="VERIFIED"
                        ):
                            events.append("STARTER_CONFIRMED")
                    elif kind=="LINEUP":
                        if prev is not None and prev.value != cur.value:
                            events.append("LINEUP_CHANGED")
                        if str(cur.state).upper()=="VERIFIED" and (
                            prev is None or str(prev.state).upper()!="VERIFIED"
                        ):
                            events.append("LINEUP_CONFIRMED")
                    elif kind=="WEATHER":
                        a=_num_weather(prev.value) if prev is not None else []
                        b=_num_weather(cur.value)
                        if prev is not None and a and b:
                            diffs=[abs(x-y) for x,y in zip(a,b) if x==x and y==y]
                            if diffs and max(diffs)>1e-6:
                                events.append("WEATHER_CHANGED")
                    elif kind=="AVAILABILITY":
                        pv=str(prev.value).upper() if prev is not None else ""
                        cv=str(cur.value).upper()
                        if cv=="PLAYER_OUT" and pv!="PLAYER_OUT":
                            events.append("PLAYER_OUT")
                        elif cv=="PLAYER_RETURNED" and pv!="PLAYER_RETURNED":
                            events.append("PLAYER_RETURNED")
                    elif kind=="REST_TRAVEL":
                        if prev is not None and prev.value != cur.value:
                            events.append("REST_TRAVEL_CHANGED")

            for event in dict.fromkeys(events):
                emitted.append({
                    "game_id":gid,
                    "prediction_time_utc":env["prediction_time_utc"],
                    "event":event,
                    "source_snapshot_ids":[
                        getattr(previous.get(k),"snapshot_id",None)
                        for k in previous.keys()
                    ] if previous else [],
                    "pit_safe":True,
                })
            previous=current

    with OUTPUT.open("w",encoding="utf-8") as fh:
        for row in emitted:
            fh.write(json.dumps(row,ensure_ascii=False,sort_keys=True)+"\n")
    print(json.dumps({
        "status":"PASS",
        "events":len(emitted),
        "games":len(by_game),
        "output":str(OUTPUT),
    },ensure_ascii=False))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
