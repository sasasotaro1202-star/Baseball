#!/usr/bin/env python3
"""Competition/postseason scope audit for Baseball prediction research.

This lane is deliberately prediction-neutral. It inventories competition
classes already present in local data and refuses to infer a class from date
alone. Unknown or missing metadata stays UNKNOWN/DEFERRED.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import requests

from npb_game_type import classify_npb_game

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "competition_scope_audit.json"
MLB_API = "https://statsapi.mlb.com/api/v1"


def audit_npb() -> dict:
    files = sorted((ROOT / "data" / "npb_games").glob("*_multi_source_pbp.csv"))
    if not files:
        return {"status": "DEFERRED", "reason": "no NPB multi-source data on current checkout"}
    frames = []
    for path in files:
        try:
            frames.append(pd.read_csv(path, low_memory=False))
        except Exception as exc:
            return {"status": "DEFERRED", "reason": f"unreadable NPB file: {path}: {exc}"}
    df = pd.concat(frames, ignore_index=True, sort=False)
    raw = df.get("game_type", pd.Series("", index=df.index)).fillna("").astype(str)
    cats = raw.map(lambda x: classify_npb_game(x)["category"])
    counts = cats.value_counts().to_dict()
    return {
        "status": "PASS",
        "rows": int(len(df)),
        "category_counts": {str(k): int(v) for k, v in counts.items()},
        "tracked_categories": [
            "regular", "interleague", "climax", "japan_series",
            "allstar", "special", "unknown", "excluded",
        ],
    }


def fetch_mlb_game_types() -> dict:
    try:
        r = requests.get(f"{MLB_API}/gameTypes", timeout=20)
        r.raise_for_status()
        payload = r.json()
        if not isinstance(payload, list):
            return {"status": "DEFERRED", "reason": "MLB gameTypes response was not a list"}
        return {
            "status": "PASS",
            "codes": {str(x.get("id")): str(x.get("description") or "") for x in payload if isinstance(x, dict)},
        }
    except Exception as exc:
        return {"status": "DEFERRED", "reason": f"MLB gameTypes endpoint unavailable: {type(exc).__name__}: {exc}"}


def audit_mlb() -> dict:
    path = ROOT / "data" / "mlb_games.csv"
    if not path.exists():
        return {"status": "DEFERRED", "reason": "no MLB cache on current checkout"}
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as exc:
        return {"status": "DEFERRED", "reason": f"unreadable MLB cache: {exc}"}
    if "game_type" not in df.columns:
        return {
            "status": "DEFERRED",
            "reason": "MLB cache predates official game_type capture; next acquisition refresh will add it",
            "rows": int(len(df)),
        }
    raw = df["game_type"].fillna("").astype(str).str.strip().replace("", "UNKNOWN")
    counts = raw.value_counts().to_dict()
    return {
        "status": "PASS",
        "rows": int(len(df)),
        "game_type_counts": {str(k): int(v) for k, v in counts.items()},
        "postseason_rows": int((raw.isin(["P", "F", "D", "L", "W"])).sum()),
        "unknown_rows": int((raw == "UNKNOWN").sum()),
    }


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "research_only": True,
        "prediction_affecting": False,
        "npb": audit_npb(),
        "mlb": audit_mlb(),
        "mlb_game_types": fetch_mlb_game_types(),
        "international": {
            "status": "DEFERRED",
            "reason": "no unverified source is inferred; add only after a free canonical source is validated",
        },
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
