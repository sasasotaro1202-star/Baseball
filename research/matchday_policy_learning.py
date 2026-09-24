#!/usr/bin/env python3
"""Adapter from the canonical Matchday OOS effect artifact to the production-safe policy schema.

The canonical learner is research/matchday_effect_fit.py. This module deliberately
does not fit a second model. It translates only an eligible OOS effect artifact
into the bounded policy representation consumed by research/matchday_policy.py.
"""
from __future__ import annotations

import json
from pathlib import Path

RESULTS = Path("results")
EFFECTS = RESULTS / "matchday_effects.json"
OUTPUT = RESULTS / "matchday_policy.json"
CLASS_NAMES = {0: "home", 1: "draw", 2: "away"}


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    if not EFFECTS.exists() or EFFECTS.stat().st_size == 0:
        payload = {
            "schema_version": 1,
            "status": "DEFERRED",
            "eligible": False,
            "reason": "canonical Matchday OOS effect artifact is missing",
            "production_auto_promotion": False,
        }
        OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(payload, ensure_ascii=False))
        return 0

    try:
        src = json.loads(EFFECTS.read_text(encoding="utf-8"))
    except Exception as exc:
        payload = {
            "schema_version": 1,
            "status": "DEFERRED",
            "eligible": False,
            "reason": f"canonical effect artifact parse failed: {exc}",
            "production_auto_promotion": False,
        }
        OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(payload, ensure_ascii=False))
        return 0

    eligible = bool(src.get("candidate_eligible")) and src.get("status") == "PASS"
    # Preserve one record per (event, class) effect. A single event can
    # legitimately move multiple NPB outcome classes; collapsing to a dict by
    # event silently discarded all but the last class.
    converted = []
    for effect in src.get("effects", []):
        try:
            key = str(effect["key"])
            coef = float(effect["coefficient"])
            cap = abs(float(effect.get("max_abs_logit", 0.15)))
            cls = int(effect["class_index"])
            if cls not in CLASS_NAMES:
                continue
            if abs(coef) < 1e-9 or cap <= 0:
                continue
            converted.append({
                "key": key,
                "class_index": cls,
                "coef": max(-cap, min(cap, coef)),
                "cap": cap,
                "target": CLASS_NAMES[cls],
                "support_rows": int(effect.get("support_rows", 0) or 0),
            })
        except Exception:
            continue

    if not eligible or not converted:
        payload = {
            "schema_version": 1,
            "status": "DEFERRED",
            "eligible": False,
            "reason": src.get("reason", "canonical Matchday effect candidate is not eligible"),
            "source_artifact": str(EFFECTS),
            "production_auto_promotion": False,
        }
    else:
        payload = {
            "schema_version": 1,
            "status": "PASS",
            "eligible": True,
            "effects": converted,
            "fusion_alpha": float(src.get("fusion_alpha", 0.0) or 0.0),
            "source_artifact": str(EFFECTS),
            "production_auto_promotion": False,
            "requires_frozen_holdout": True,
            "requires_integrity_gate": True,
        }

    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
