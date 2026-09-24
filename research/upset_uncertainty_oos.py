#!/usr/bin/env python3
"""Research-only upset/surprise-risk and uncertainty layer.

The layer estimates the probability that the model's current favorite will be
wrong, then tests a bounded confidence shrinkage candidate. It never reverses
the predicted side and never changes production automatically.

PIT rule:
- Features are derived only from probabilities available before the outcome.
- Expert disagreement is computed from current expert probabilities only.
- The outcome is used only after the risk model is fit and the candidate forecast
  is produced for the current chronological evaluation row.

"Upset" is defined conservatively here as: actual outcome != model favorite.
This is a surprise/error event, not a claim about betting-market underdogs.
"""
from __future__ import annotations
import json, math
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MIN_ROWS = 180
FIT_FRAC = 0.60
TUNE_FRAC = 0.15
MIN_TUNE = 30
MIN_EVAL = 50
BETA_GRID = np.linspace(0.0, 1.50, 61)
EPS = 1e-12

def _prob_cols(league: str) -> List[str]:
    return ["pred_home", "pred_draw", "pred_away"] if league == "NPB" else ["pred_home", "pred_away"]

def _normalize(p: np.ndarray) -> np.ndarray:
    x = np.asarray(p, dtype=float)
    if x.ndim != 2 or x.shape[1] < 2:
        raise ValueError("probability matrix must be 2-D with at least two classes")
    if not np.all(np.isfinite(x)):
        raise ValueError("non-finite probabilities")
    x = np.clip(x, EPS, 1.0)
    return x / np.maximum(x.sum(axis=1, keepdims=True), EPS)

def _metrics(p: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    p = _normalize(p); y = np.asarray(y, dtype=int); idx = np.arange(len(y))
    ll = float(-np.mean(np.log(np.clip(p[idx, y], EPS, 1.0))))
    target = np.zeros_like(p); target[idx, y] = 1.0
    brier = float(np.mean(np.sum((p - target) ** 2, axis=1)))
    pred = np.argmax(p, axis=1); accuracy = float(np.mean(pred == y))
    conf = np.max(p, axis=1); edges = np.linspace(0.0, 1.0, 11); ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf >= lo) & (conf <= hi if hi == 1.0 else conf < hi)
        if np.any(mask):
            ece += float(mask.mean()) * abs(float(conf[mask].mean()) - float((pred[mask] == y[mask]).mean()))
    return {"logloss": ll, "brier": brier, "accuracy": accuracy, "ece": float(ece)}

def _expert_disagreement(df: pd.DataFrame, league: str) -> np.ndarray:
    experts = sorted({c[len("expert_"):].rsplit("_", 1)[0] for c in df.columns if c.startswith("expert_") and c.endswith("_home")})
    if len(experts) < 2: return np.zeros(len(df), dtype=float)
    matrices = []
    for key in experts:
        cols = [f"expert_{key}_home"] + ([f"expert_{key}_draw", f"expert_{key}_away"] if league == "NPB" else [f"expert_{key}_away"])
        if any(c not in df.columns for c in cols): continue
        arr = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        if np.all(np.isfinite(arr)): matrices.append(arr)
    if len(matrices) < 2: return np.zeros(len(df), dtype=float)
    cube = np.stack([_normalize(x) for x in matrices], axis=1)
    return np.clip(np.mean(np.std(cube, axis=1), axis=1) / 0.25, 0.0, 1.0)

def _risk_features(p: np.ndarray, disagreement: np.ndarray) -> np.ndarray:
    p = _normalize(p); top = np.max(p, axis=1)
    order = np.argsort(p, axis=1)[:, ::-1]
    second = p[np.arange(len(p)), order[:, 1]]
    margin = top - second
    entropy = -np.sum(p * np.log(np.clip(p, EPS, 1.0)), axis=1) / max(math.log(p.shape[1]), EPS)
    return np.column_stack([1.0-top, 1.0-margin/np.maximum(top, EPS), entropy, np.asarray(disagreement, dtype=float)])

def _fit_error_model(x: np.ndarray, y: np.ndarray) -> Optional[LogisticRegression]:
    labels = np.asarray(y, dtype=int)
    if len(labels) < 60 or np.unique(labels).size < 2: return None
    model = LogisticRegression(C=0.75, class_weight="balanced", max_iter=2000, random_state=42)
    model.fit(np.asarray(x, dtype=float), labels)
    return model

def _adjust_probabilities(p: np.ndarray, risk: np.ndarray, beta: float) -> np.ndarray:
    pp = _normalize(p)
    temp = 1.0 + float(np.clip(beta, 0.0, 1.5)) * np.clip(risk, 0.0, 1.0)
    out = np.vstack([np.power(np.clip(row, EPS, 1.0), 1.0/float(t)) for row,t in zip(pp,temp)])
    return _normalize(out)

def _high_confidence_error(p: np.ndarray, y: np.ndarray, threshold: float = 0.70) -> float:
    p = _normalize(p); y = np.asarray(y, dtype=int); pred = np.argmax(p, axis=1)
    mask = np.max(p, axis=1) >= threshold
    return float(np.mean(pred[mask] != y[mask])) if np.any(mask) else float("nan")

def _evaluate(league: str, df: pd.DataFrame) -> Dict[str, object]:
    cols = _prob_cols(league)
    if any(c not in df.columns for c in cols) or "actual" not in df.columns:
        return {"status":"DEFERRED","reason":"required OOS probability columns are missing","rows":0}
    if "datetime" in df.columns:
        sort_cols = ["datetime","game_id"] if "game_id" in df.columns else ["datetime"]
        df = df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    else: df = df.reset_index(drop=True)
    probs = df[cols].apply(pd.to_numeric, errors="coerce"); actual = pd.to_numeric(df["actual"], errors="coerce")
    valid = probs.notna().all(axis=1) & actual.notna()
    df = df.loc[valid].copy(); probs = probs.loc[valid]; actual = actual.loc[valid].astype(int)
    if len(df) < MIN_ROWS:
        return {"status":"DEFERRED","reason":f"insufficient OOS rows: {len(df)} < {MIN_ROWS}","rows":int(len(df))}
    base = _normalize(probs.to_numpy(dtype=float)); y = actual.to_numpy(dtype=int)
    if np.any(y < 0) or np.any(y >= base.shape[1]):
        return {"status":"DEFERRED","reason":"actual labels outside probability columns","rows":int(len(df))}
    disagreement = _expert_disagreement(df, league); features = _risk_features(base, disagreement)
    n = len(df); fit_end=max(60,int(math.floor(n*FIT_FRAC)))
    tune_end=min(n-MIN_EVAL,max(fit_end+MIN_TUNE,int(math.floor(n*(FIT_FRAC+TUNE_FRAC)))))
    if tune_end <= fit_end or n-tune_end < MIN_EVAL:
        return {"status":"DEFERRED","reason":"chronological fit/tune/eval split is too small","rows":int(n)}
    base_pred=np.argmax(base,axis=1); error_target=(base_pred!=y).astype(int)
    risk_model=_fit_error_model(features[:fit_end],error_target[:fit_end])
    if risk_model is None:
        return {"status":"DEFERRED","reason":"error-risk model lacks both classes in fit segment","rows":int(n)}
    tune_risk=np.clip(risk_model.predict_proba(features[fit_end:tune_end])[:,1],0.0,1.0)
    eval_risk=np.clip(risk_model.predict_proba(features[tune_end:])[:,1],0.0,1.0)
    tune_base=base[fit_end:tune_end]; tune_y=y[fit_end:tune_end]
    eval_base=base[tune_end:]; eval_y=y[tune_end:]
    best_beta=0.0; best_loss=float("inf"); beta_results=[]
    for beta in BETA_GRID:
        cand=_adjust_probabilities(tune_base,tune_risk,float(beta)); m=_metrics(cand,tune_y)
        beta_results.append({"beta":float(beta),**m})
        if m["logloss"] < best_loss-1e-12: best_loss=m["logloss"]; best_beta=float(beta)
    eval_candidate=_adjust_probabilities(eval_base,eval_risk,best_beta)
    eval_base_metrics=_metrics(eval_base,eval_y); eval_candidate_metrics=_metrics(eval_candidate,eval_y)
    risk_threshold=float(np.quantile(tune_risk,0.75)); high_mask=eval_risk>=risk_threshold
    high_case={
      "threshold":risk_threshold,"rows":int(high_mask.sum()),
      "baseline_error_rate":float(np.mean(np.argmax(eval_base[high_mask],axis=1)!=eval_y[high_mask])) if np.any(high_mask) else float("nan"),
      "candidate_error_rate":float(np.mean(np.argmax(eval_candidate[high_mask],axis=1)!=eval_y[high_mask])) if np.any(high_mask) else float("nan"),
      "mean_risk":float(np.mean(eval_risk[high_mask])) if np.any(high_mask) else float("nan")}
    dll=float(eval_candidate_metrics["logloss"]-eval_base_metrics["logloss"]); db=float(eval_candidate_metrics["brier"]-eval_base_metrics["brier"])
    da=float(eval_candidate_metrics["accuracy"]-eval_base_metrics["accuracy"]); de=float(eval_candidate_metrics["ece"]-eval_base_metrics["ece"])
    rel_ll=float(-dll/max(eval_base_metrics["logloss"],EPS)); rel_brier=float(-db/max(eval_base_metrics["brier"],EPS))
    baseline_hc=_high_confidence_error(eval_base,eval_y); candidate_hc=_high_confidence_error(eval_candidate,eval_y)
    high_case_safe=(not np.isfinite(high_case["baseline_error_rate"]) or high_case["candidate_error_rate"]<=high_case["baseline_error_rate"])
    eligible=bool(rel_ll>=0.03 and rel_brier>=0.01 and da>=-0.005 and de<=0.0+1e-12 and high_case_safe)
    return {
      "status":"PASS","league":league,"rows":int(n),"fit_rows":int(fit_end),"tune_rows":int(tune_end-fit_end),"evaluation_rows":int(n-tune_end),
      "risk_features":["one_minus_top_probability","one_minus_top_margin_ratio","normalized_entropy","expert_disagreement_normalized"],
      "upset_definition":"actual outcome != model favorite","fit_error_rate":float(error_target[:fit_end].mean()),
      "tune_error_rate":float(error_target[fit_end:tune_end].mean()),"eval_error_rate":float(error_target[tune_end:].mean()),
      "tune_best_beta":float(best_beta),"tune_best_logloss":float(best_loss),"baseline":eval_base_metrics,"candidate":eval_candidate_metrics,
      "relative_logloss_improvement":rel_ll,"relative_brier_improvement":rel_brier,"delta_accuracy":da,"delta_ece":de,
      "high_uncertainty_case":high_case,"high_confidence_error_rate_baseline":baseline_hc,
      "high_confidence_error_rate_candidate":candidate_hc,"candidate_eligible":eligible,"production_auto_promotion":False,
      "prediction_effect":"confidence_shrink_only; never reverses the model favorite","beta_search_summary":beta_results[:5]+beta_results[-5:],
    }

def main() -> int:
    RESULTS.mkdir(parents=True,exist_ok=True)
    payload={"schema_version":1,"artifact_type":"OOS_UPSET_UNCERTAINTY_RISK","status":"DEFERRED","production_auto_promotion":False,"leagues":{}}
    for league in ("NPB","MLB"):
        path=RESULTS/f"{league.lower()}_backtest_results.csv"
        if not path.exists() or path.stat().st_size==0:
            payload["leagues"][league]={"status":"DEFERRED","reason":"OOS result file missing","rows":0}; continue
        try: payload["leagues"][league]=_evaluate(league,pd.read_csv(path))
        except Exception as exc: payload["leagues"][league]={"status":"DEFERRED","reason":f"risk evaluation failed safely: {type(exc).__name__}: {exc}","rows":0}
    if any(v.get("status")=="PASS" for v in payload["leagues"].values()): payload["status"]="PASS"
    (RESULTS/"upset_uncertainty_oos.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(payload,ensure_ascii=False,sort_keys=True)); return 0

if __name__=="__main__": raise SystemExit(main())
