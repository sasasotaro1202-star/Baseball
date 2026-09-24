#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Research-only chronological probability calibration challenger."""
from __future__ import annotations
import json, math
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
ROOT=Path(__file__).resolve().parents[1]; RESULTS=ROOT/"results"; MIN_ROWS=120; CAL_FRACTION=0.70; MAX_ECE=0.020

def _clip_normalize(p: np.ndarray) -> np.ndarray:
    p=np.asarray(p,dtype=float)
    p=np.nan_to_num(p,nan=1.0/p.shape[1],posinf=1.0/p.shape[1],neginf=1.0/p.shape[1])
    p=np.clip(p,1e-7,1.0); return p/p.sum(axis=1,keepdims=True)

def _ece(p: np.ndarray,y: np.ndarray,bins:int=10)->float:
    conf=np.max(p,axis=1); pred=np.argmax(p,axis=1); edges=np.linspace(0.0,1.0,bins+1); score=0.0
    for lo,hi in zip(edges[:-1],edges[1:]):
        mask=(conf>=lo)&(conf<(hi if hi<1.0 else hi+1e-12))
        if np.any(mask): score+=float(mask.mean())*abs(float(conf[mask].mean())-float((pred[mask]==y[mask]).mean()))
    return float(score)

def _metrics(p:np.ndarray,y:np.ndarray)->Dict[str,float]:
    p=_clip_normalize(p); idx=np.arange(len(y)); ll=float(-np.mean(np.log(np.clip(p[idx,y],1e-7,1.0))))
    target=np.zeros_like(p); target[idx,y]=1.0
    return {"logloss":ll,"brier":float(np.mean(np.sum((p-target)**2,axis=1))),"ece":_ece(p,y)}

def _temperature_fit(p:np.ndarray,y:np.ndarray)->float:
    best_t,best_ll=1.0,float("inf")
    for t in np.linspace(0.65,2.25,65):
        ll=_metrics(np.power(np.clip(p,1e-7,1.0),1.0/t),y)["logloss"]
        if ll<best_ll: best_ll,best_t=ll,float(t)
    return best_t

def _platt_fit(p:np.ndarray,y:np.ndarray):
    k=p.shape[1]; models=[]; out=np.zeros_like(p)
    logits=np.log(np.clip(p,1e-6,1-1e-6)/np.clip(1-p,1e-6,1.0))
    for c in range(k):
        target=(y==c).astype(int)
        if np.unique(target).size<2: models.append(None); out[:,c]=float(target.mean()); continue
        m=LogisticRegression(C=1.0,max_iter=1000,random_state=42); m.fit(logits[:,[c]],target); models.append(m)
        out[:,c]=m.predict_proba(logits[:,[c]])[:,1]
    return models,_clip_normalize(out)

def _platt_apply(models,p):
    logits=np.log(np.clip(p,1e-6,1-1e-6)/np.clip(1-p,1e-6,1.0)); out=np.zeros_like(p); k=p.shape[1]
    for c,m in enumerate(models): out[:,c]=1.0/k if m is None else m.predict_proba(logits[:,[c]])[:,1]
    return _clip_normalize(out)

def _isotonic_fit(p,y):
    models=[]; out=np.zeros_like(p)
    for c in range(p.shape[1]):
        target=(y==c).astype(float)
        if np.unique(target).size<2: models.append(None); out[:,c]=float(target.mean()); continue
        m=IsotonicRegression(out_of_bounds="clip",y_min=0.0,y_max=1.0); m.fit(p[:,c],target); models.append(m); out[:,c]=m.predict(p[:,c])
    return models,_clip_normalize(out)

def _isotonic_apply(models,p):
    out=np.zeros_like(p); k=p.shape[1]
    for c,m in enumerate(models): out[:,c]=1.0/k if m is None else m.predict(p[:,c])
    return _clip_normalize(out)

def _prepare(df:pd.DataFrame,league:str)->Tuple[np.ndarray,np.ndarray]:
    cols=["pred_home","pred_draw","pred_away"] if league=="NPB" else ["pred_home","pred_away"]
    if any(c not in df.columns for c in cols) or "actual" not in df.columns: return np.empty((0,len(cols))),np.empty(0,dtype=int)
    x=df[cols].apply(pd.to_numeric,errors="coerce"); y=pd.to_numeric(df["actual"],errors="coerce"); ok=x.notna().all(axis=1)&y.notna()
    if not ok.any(): return np.empty((0,len(cols))),np.empty(0,dtype=int)
    return _clip_normalize(x.loc[ok].to_numpy(float)),y.loc[ok].to_numpy(int)

def evaluate_league(league:str)->Dict[str,object]:
    path=RESULTS/f"{league.lower()}_backtest_results.csv"
    if not path.exists(): return {"status":"DEFERRED","reason":"OOS result file missing","rows":0}
    try: df=pd.read_csv(path)
    except Exception as exc: return {"status":"DEFERRED","reason":f"OOS result unreadable: {exc}","rows":0}
    if len(df)<MIN_ROWS: return {"status":"DEFERRED","reason":f"insufficient OOS rows: {len(df)} < {MIN_ROWS}","rows":int(len(df))}
    if "datetime" in df.columns: df=df.sort_values(["datetime","game_id"] if "game_id" in df.columns else ["datetime"],kind="mergesort")
    p,y=_prepare(df,league)
    if len(y)<MIN_ROWS: return {"status":"DEFERRED","reason":"insufficient valid probability rows","rows":int(len(y))}
    cut=min(max(int(math.floor(len(y)*CAL_FRACTION)),60),len(y)-30)
    p_cal,y_cal,p_eval,y_eval=p[:cut],y[:cut],p[cut:],y[cut:]
    methods={"raw":_metrics(p_eval,y_eval)}
    t=_temperature_fit(p_cal,y_cal); methods["temperature"]=_metrics(np.power(np.clip(p_eval,1e-7,1.0),1.0/t),y_eval)
    iso_models,_=_isotonic_fit(p_cal,y_cal); methods["isotonic"]=_metrics(_isotonic_apply(iso_models,p_eval),y_eval)
    platt_models,_=_platt_fit(p_cal,y_cal); methods["platt"]=_metrics(_platt_apply(platt_models,p_eval),y_eval)
    raw=methods["raw"]; eligible=[n for n,m in methods.items() if n!="raw" and m["logloss"]<raw["logloss"] and m["brier"]<=raw["brier"] and m["ece"]<=raw["ece"]+MAX_ECE]
    best=min(["raw"]+eligible,key=lambda n:(methods[n]["logloss"],methods[n]["brier"],methods[n]["ece"]))
    return {"status":"PASS","league":league,"rows":int(len(y)),"calibration_rows":int(len(y_cal)),"evaluation_rows":int(len(y_eval)),"temperature":float(t),"methods":methods,"candidate_eligible":bool(best!="raw"),"candidate_method":best,"promotion_auto":False,"selection_rule":"later untouched chronological OOS must improve LogLoss, not worsen Brier, and keep ECE within bounded tolerance"}

def main():
    payload={"schema_version":1,"status":"DEFERRED","production_auto_promotion":False,"split":{"fit_fraction":CAL_FRACTION,"evaluation_fraction":1.0-CAL_FRACTION},"leagues":{l:evaluate_league(l) for l in ("NPB","MLB")}}
    if any(v.get("status")=="PASS" for v in payload["leagues"].values()): payload["status"]="PASS"
    RESULTS.mkdir(parents=True,exist_ok=True); (RESULTS/"calibration_oos.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"); print(json.dumps(payload,ensure_ascii=False))
if __name__=="__main__": main()
