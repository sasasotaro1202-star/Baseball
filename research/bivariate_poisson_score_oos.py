#!/usr/bin/env python3
"""Chronological OOS bivariate-Poisson score challenger.

This research-only candidate adds a shared scoring component so home/away
runs may be correlated. It compares directly against the incumbent independent
Poisson distribution on the user-facing score-choice and Low/High events.
No production state is changed and no automatic promotion is allowed.
"""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
import pandas as pd

RESULTS = Path("results")
MIN_ROWS = 160
FIT_FRAC = 0.60
MIN_SCORE_LL_IMPROVEMENT = 0.001
MAX_TOP4_REGRESSION = 0.01
MAX_LOW_HIGH_BRIER_REGRESSION = 0.005

def poisson_pmf(k, mu):
    mu = max(float(mu), 1e-9)
    return math.exp(-mu + int(k) * math.log(mu) - math.lgamma(int(k) + 1.0))

def _log_term(h, a, k, l1, l2, l3):
    vals = []
    if h-k: vals.append((h-k) * math.log(l1) - math.lgamma(h-k+1.0))
    if a-k: vals.append((a-k) * math.log(l2) - math.lgamma(a-k+1.0))
    if k: vals.append(k * math.log(l3) - math.lgamma(k+1.0))
    return sum(vals)

def bivariate_pmf(h, a, mu_h, mu_a, shared):
    mu_h = max(float(mu_h), 1e-9)
    mu_a = max(float(mu_a), 1e-9)
    shared = float(np.clip(shared, 0.0, min(mu_h, mu_a)))
    l1, l2, l3 = mu_h-shared, mu_a-shared, shared
    if l3 <= 1e-12:
        return poisson_pmf(h, mu_h) * poisson_pmf(a, mu_a)
    base = -(l1+l2+l3)
    terms = [base + _log_term(h,a,k,l1,l2,l3) for k in range(min(h,a)+1)]
    m = max(terms)
    return math.exp(m) * sum(math.exp(x-m) for x in terms)

def score_space(mu_h, mu_a, shared, max_run=18):
    cells = []
    for h in range(max_run+1):
        for a in range(max_run+1):
            cells.append((f"{h}-{a}", bivariate_pmf(h,a,mu_h,mu_a,shared)))
    p_other = max(0.0, 1.0 - sum(p for s,p in cells if int(s.split("-",1)[0]) <= 6 and int(s.split("-",1)[1]) <= 6))
    cells = [(s,p) for s,p in cells if int(s.split("-",1)[0]) <= 6 and int(s.split("-",1)[1]) <= 6]
    cells.append(("その他", p_other))
    total = sum(p for _,p in cells)
    return [(s, float(p/total)) for s,p in cells]

def top4_from_space(space):
    return sorted(
        space,
        key=lambda z: (
            -float(z[1]),
            1 if z[0] == "その他" else 0,
            (999, 999) if z[0] == "その他" else tuple(map(int, z[0].split("-", 1))),
        ),
    )[:4]

def low_high_probs_bivariate(mu_h, mu_a, shared):
    max_run = max(18, int(math.ceil(max(float(mu_h), float(mu_a)) * 4.0 + 8.0)))
    low = 0.0
    total = 0.0
    for h in range(max_run+1):
        for a in range(max_run+1):
            p = bivariate_pmf(h,a,mu_h,mu_a,shared)
            total += p
            if h+a <= 6:
                low += p
    low = float(np.clip(low/max(total,1e-12),0.0,1.0))
    return low, 1.0-low

def categorical_nll(space, actual_h, actual_a):
    target = "その他" if actual_h >= 7 or actual_a >= 7 else f"{int(actual_h)}-{int(actual_a)}"
    probs = dict(space)
    return float(-math.log(max(float(probs.get(target, 1e-12)),1e-12)))

def binary_metrics(p_high, actual_high):
    p = float(np.clip(p_high,1e-9,1-1e-9))
    y = int(actual_high)
    ll = -(y*math.log(p) + (1-y)*math.log(1-p))
    br = (p-y)**2
    return ll, br

def fit_shared(yh, mu_h, ya, mu_a):
    rh = np.asarray(yh,float) - np.asarray(mu_h,float)
    ra = np.asarray(ya,float) - np.asarray(mu_a,float)
    cov = float(np.mean((rh-rh.mean())*(ra-ra.mean())))
    upper = float(max(0.0, min(np.median(mu_h), np.median(mu_a))*0.85))
    return float(np.clip(cov, 0.0, upper))

def evaluate(path):
    try: df = pd.read_csv(path)
    except Exception as exc: return {"status":"DEFERRED","reason":f"unreadable: {exc}","rows":0}
    required={"datetime","home_score","away_score","lambda_home","lambda_away"}
    if not required.issubset(df.columns): return {"status":"DEFERRED","reason":"required score columns missing","rows":int(len(df))}
    df=df.copy()
    df["datetime"]=pd.to_datetime(df["datetime"],errors="coerce",utc=True)
    for c in required-{"datetime"}: df[c]=pd.to_numeric(df[c],errors="coerce")
    df=df.dropna(subset=list(required)).sort_values(["datetime"]+([ "game_id"] if "game_id" in df.columns else []),kind="mergesort").reset_index(drop=True)
    df=df[(df.home_score>=0)&(df.away_score>=0)&(df.lambda_home>0)&(df.lambda_away>0)]
    if len(df)<MIN_ROWS: return {"status":"DEFERRED","reason":f"insufficient rows: {len(df)} < {MIN_ROWS}","rows":int(len(df))}
    cut=min(max(int(len(df)*FIT_FRAC),80),len(df)-50)
    fit,test=df.iloc[:cut],df.iloc[cut:]
    shared=fit_shared(fit.home_score,fit.lambda_home,fit.away_score,fit.lambda_away)
    base_score_ll=[]; cand_score_ll=[]; base_top4=[]; cand_top4=[]; base_lowhigh_ll=[]; cand_lowhigh_ll=[]; base_lowhigh_br=[]; cand_lowhigh_br=[]
    for r in test.itertuples(index=False):
        mh,ma=float(r.lambda_home),float(r.lambda_away)
        target_other=float(r.home_score)>=7 or float(r.away_score)>=7
        base_space=[(f"{h}-{a}",poisson_pmf(h,mh)*poisson_pmf(a,ma)) for h in range(7) for a in range(7)]
        p_other=max(0.0,1.0-sum(p for _,p in base_space))
        base_space.append(("その他",p_other))
        z=sum(p for _,p in base_space); base_space=[(s,p/z) for s,p in base_space]
        cand_space=score_space(mh,ma,shared)
        base_top=set(s for s,_ in top4_from_space(base_space)); cand_top=set(s for s,_ in top4_from_space(cand_space))
        actual=f"{int(r.home_score)}-{int(r.away_score)}"
        base_score_ll.append(categorical_nll(base_space,r.home_score,r.away_score)); cand_score_ll.append(categorical_nll(cand_space,r.home_score,r.away_score))
        base_top4.append(int(("その他" if target_other else actual) in base_top)); cand_top4.append(int(("その他" if target_other else actual) in cand_top))
        bl,bh=(sum(poisson_pmf(k,mh+ma) for k in range(7)),0.0); bh=1.0-bl
        cl,ch=low_high_probs_bivariate(mh,ma,shared)
        actual_high = (float(r.home_score) + float(r.away_score)) >= 7
        base_ll, base_br = binary_metrics(bh, actual_high)
        cand_ll, cand_br = binary_metrics(ch, actual_high)
        base_lowhigh_ll.append(base_ll); cand_lowhigh_ll.append(cand_ll)
        base_lowhigh_br.append(base_br); cand_lowhigh_br.append(cand_br)
    base_ll=float(np.mean(base_score_ll)); cand_ll=float(np.mean(cand_score_ll))
    base_t4=float(np.mean(base_top4)); cand_t4=float(np.mean(cand_top4))
    base_hll=float(np.mean(base_lowhigh_ll)); cand_hll=float(np.mean(cand_lowhigh_ll))
    base_hbr=float(np.mean(base_lowhigh_br)); cand_hbr=float(np.mean(cand_lowhigh_br))
    eligible=bool((base_ll-cand_ll)>=MIN_SCORE_LL_IMPROVEMENT and (cand_t4-base_t4)>=-MAX_TOP4_REGRESSION and (cand_hbr-base_hbr)<=MAX_LOW_HIGH_BRIER_REGRESSION)
    return {"status":"PASS","rows":int(len(df)),"fit_rows":int(len(fit)),"final_oos_rows":int(len(test)),"shared_lambda3":shared,"independent_score_nll":base_ll,"bivariate_score_nll":cand_ll,"delta_score_nll":cand_ll-base_ll,"independent_top4_hit_rate":base_t4,"bivariate_top4_hit_rate":cand_t4,"delta_top4_hit_rate":cand_t4-base_t4,"independent_low_high_logloss":base_hll,"bivariate_low_high_logloss":cand_hll,"independent_low_high_brier":base_hbr,"bivariate_low_high_brier":cand_hbr,"candidate_eligible":eligible,"production_auto_promotion":False}

def main():
    RESULTS.mkdir(parents=True,exist_ok=True)
    payload={"schema_version":1,"artifact_type":"BIVARIATE_POISSON_SCORE_OOS","status":"DEFERRED","production_auto_promotion":False,"leagues":{}}
    for league in ("NPB","MLB"):
        path=RESULTS/f"{league.lower()}_backtest_results.csv"
        payload["leagues"][league]=evaluate(path) if path.exists() else {"status":"DEFERRED","reason":"OOS result file missing","rows":0}
    payload["status"]="PASS" if any(v.get("status")=="PASS" for v in payload["leagues"].values()) else "DEFERRED"
    (RESULTS/"bivariate_poisson_score_oos.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(payload,ensure_ascii=False,sort_keys=True))
    return 0

if __name__=="__main__": raise SystemExit(main())