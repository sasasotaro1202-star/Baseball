#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Summarize NPB game-type OOS ablation results.

A variant is defined by NPB_TRAINING_GAME_CATEGORIES.  The evaluation universe
remains all requested competitive/special categories so each incremental class
can be assessed on the same chronological evaluation population.
No promotion is performed here.
"""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/"results"
DEFAULT_EVAL=("regular","interleague","climax","japan_series","allstar","special")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--variant",default=os.getenv("ABLATON_VARIANT","manual"))
    args=p.parse_args()
    summary=RESULTS/"npb_backtest_summary.csv"
    detail=RESULTS/"npb_game_type_metrics.csv"
    out=RESULTS/"npb_game_type_ablation_summary.json"
    payload={
        "schema_version":1,
        "variant":args.variant,
        "training_categories":[x.strip() for x in os.getenv("NPB_TRAINING_GAME_CATEGORIES","").split(",") if x.strip()],
        "evaluation_categories":[x.strip() for x in os.getenv("NPB_EVALUATION_GAME_CATEGORIES",",".join(DEFAULT_EVAL)).split(",") if x.strip()],
        "promotion_auto":False,
        "status":"DEFERRED",
        "reason":"OOS result files not available",
    }
    if summary.exists() and detail.exists():
        try:
            s=pd.read_csv(summary)
            d=pd.read_csv(detail)
            payload["status"]="PASS"
            payload["summary"]=s.iloc[-1].to_dict() if not s.empty else {}
            payload["by_category"]=d.to_dict("records")
            payload["sample_counts"]={str(k):int(v) for k,v in d.groupby("Category").size().items()} if "Category" in d else {}
            payload["reason"]=""
        except Exception as exc:
            payload["status"]="FAIL"
            payload["reason"]=f"cannot parse OOS results: {exc}"
            raise
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(payload,ensure_ascii=False,indent=2,default=str)+"\n",encoding="utf-8")
    print(json.dumps(payload,ensure_ascii=False))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
