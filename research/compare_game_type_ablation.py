#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare game-type OOS variants and issue a research-only adoption verdict."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
ROOT_OUT=ROOT/"results"


def load_all(root: Path):
    rows=[]
    for p in root.rglob("npb_game_type_ablation_summary.json"):
        try:
            d=json.loads(p.read_text(encoding="utf-8"))
            d["_source"]=str(p)
            rows.append(d)
        except Exception:
            continue
    return rows


def main():
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",default=".")
    args=ap.parse_args()
    rows=load_all(Path(args.root))
    out={"schema_version":1,"status":"DEFERRED","promotion_auto":False,"variants":rows,"verdict":"DEFERRED"}
    passed=[x for x in rows if x.get("status")=="PASS" and isinstance(x.get("summary"),dict)]
    base=next((x for x in passed if x.get("variant")=="baseline"),None)
    if base:
        bs=base["summary"]
        comparable=[]
        for x in passed:
            s=x["summary"]
            if x is base: continue
            ll=float(s.get("LogLoss",float("nan")))
            bll=float(bs.get("LogLoss",float("nan")))
            acc=float(s.get("Accuracy",float("nan")))
            bacc=float(bs.get("Accuracy",float("nan")))
            if ll==ll and bll==bll:
                comparable.append({
                    "variant":x.get("variant"),
                    "logloss_delta":ll-bll,
                    "logloss_relative_improvement":(bll-ll)/max(abs(bll),1e-9),
                    "accuracy_delta":acc-bacc if acc==acc and bacc==bacc else None,
                })
        out["status"]="PASS"
        out["comparisons"]=comparable
        out["verdict"]="RESEARCH_ONLY"
        out["adoption_rule"]="Require fixed OOS/frozen-holdout/calibration/PIT gates; no automatic promotion."
    ROOT_OUT.mkdir(parents=True,exist_ok=True)
    (ROOT_OUT/"npb_game_type_ablation_comparison.json").write_text(
        json.dumps(out,ensure_ascii=False,indent=2,default=str)+"\n",encoding="utf-8")
    print(json.dumps(out,ensure_ascii=False))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
