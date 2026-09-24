#!/usr/bin/env python3
"""Research-only rolling split-conformal uncertainty for multiclass forecasts.

This module does not alter probabilities. It provides a prediction set and a
simple uncertainty diagnostic using past resolved rows only. Under distribution
shift, coverage is not guaranteed; the caller must therefore treat the result
as risk information, not a correctness certificate.
"""
from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np


def nonconformity(probabilities: np.ndarray, outcomes: np.ndarray) -> np.ndarray:
    p=np.asarray(probabilities,dtype=float)
    y=np.asarray(outcomes,dtype=int).reshape(-1)
    if p.ndim!=2 or len(p)!=len(y) or not np.all(np.isfinite(p)):
        raise ValueError("invalid conformal history")
    if np.any(y<0) or np.any(y>=p.shape[1]):
        raise ValueError("outcomes outside probability columns")
    p=np.clip(p,1e-12,1.0)
    p/=p.sum(axis=1,keepdims=True)
    return 1.0-p[np.arange(len(y)),y]


def class_pvalues(probability: np.ndarray, history_probabilities: np.ndarray, history_outcomes: np.ndarray) -> np.ndarray:
    q=np.asarray(probability,dtype=float).reshape(-1)
    h=np.asarray(history_probabilities,dtype=float)
    y=np.asarray(history_outcomes,dtype=int).reshape(-1)
    if h.ndim!=2 or len(h)!=len(y) or h.shape[1]!=len(q) or len(y)==0:
        raise ValueError("conformal history shape mismatch")
    q=np.clip(q,1e-12,1.0); q/=q.sum()
    scores=nonconformity(h,y)
    current=1.0-q
    # Conservative finite-sample smoothed p-value.
    return np.asarray([
        (1.0+float(np.sum(scores>=c-1e-12)))/(len(scores)+1.0)
        for c in current
    ],dtype=float)


def prediction_set(
    probability: np.ndarray,
    history_probabilities: np.ndarray,
    history_outcomes: np.ndarray,
    alpha: float=0.10,
) -> Tuple[np.ndarray, np.ndarray]:
    p=np.asarray(probability,dtype=float).reshape(-1)
    pv=class_pvalues(p,history_probabilities,history_outcomes)
    a=float(np.clip(alpha,0.01,0.50))
    return np.flatnonzero(pv>a), pv


def uncertainty_summary(
    probability: np.ndarray,
    history_probabilities: np.ndarray,
    history_outcomes: np.ndarray,
    alpha: float=0.10,
) -> Dict[str, object]:
    p=np.asarray(probability,dtype=float).reshape(-1)
    p=np.clip(p,1e-12,1.0); p/=p.sum()
    pred_set,pv=prediction_set(p,history_probabilities,history_outcomes,alpha)
    order=np.argsort(p)[::-1]
    top=float(p[order[0]])
    second=float(p[order[1]]) if len(p)>1 else 0.0
    return {
        "prediction_set":[int(x) for x in pred_set],
        "prediction_set_size":int(len(pred_set)),
        "class_pvalues":[float(x) for x in pv],
        "top_probability":top,
        "top_gap":float(top-second),
        "entropy":float(-np.sum(p*np.log(p))),
        "alpha":float(np.clip(alpha,0.01,0.50)),
        "history_rows":int(len(history_outcomes)),
    }


def self_test() -> dict:
    p=np.array([[0.80,0.15,0.05],[0.78,0.17,0.05],[0.82,0.12,0.06],[0.75,0.20,0.05]])
    y=np.array([0,0,0,1])
    cur=np.array([0.84,0.10,0.06])
    s=uncertainty_summary(cur,p,y,alpha=0.10)
    assert s["prediction_set_size"]>=1
    assert len(s["class_pvalues"])==3
    assert abs(sum(cur)-1.0)<1e-9
    return {"status":"PASS","summary":s}


if __name__=="__main__":
    import json
    print(json.dumps(self_test(),ensure_ascii=False,sort_keys=True))
