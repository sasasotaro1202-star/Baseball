#!/usr/bin/env python3
"""Chronological OOS learner for bounded Matchday Intelligence effects.

Expected checkpoint columns:
  pred_home / pred_draw / pred_away
  ctx_<EVENT_NAME>  (0/1 event indicator)
  actual
  datetime / game_id

A promotable artifact is emitted only when the learned context layer improves
a later chronological validation segment on LogLoss and Brier without material
Accuracy/ECE regression. Otherwise the output remains DEFERRED.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path("results")
CHECKPOINTS = Path("data/checkpoints")
OUTPUT = RESULTS / "matchday_effects.json"

MIN_ROWS = 160
MIN_EVENT_ROWS = 25
SHRINKAGE = 20.0
MAX_ABS_LOGIT = 0.15
MIN_LL = 0.0005
MIN_BRIER = 0.00025
MAX_ACC_REG = 0.005
MAX_ECE_REG = 0.010


def normalize(p):
    x = np.asarray(p, dtype=float)
    x = np.clip(x, 1e-8, 1.0)
    return x / x.sum(axis=1, keepdims=True)


def metrics(p, y):
    p = normalize(p)
    y = np.asarray(y, dtype=int)
    ll = float(np.mean(-np.log(p[np.arange(len(y)), y])))
    target = np.zeros_like(p)
    target[np.arange(len(y),), y] = 1.0
    br = float(np.mean(np.sum((p - target) ** 2, axis=1)))
    acc = float(np.mean(p.argmax(axis=1) == y))
    conf = p.max(axis=1)
    pred = p.argmax(axis=1)
    err = (pred != y).astype(float)
    ece = 0.0
    edges = np.linspace(0.0, 1.0, 11)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf >= lo) & (conf < hi if hi < 1.0 else conf <= hi)
        if mask.any():
            ece += float(mask.mean()) * abs(float(conf[mask].mean()) - float(1.0 - err[mask].mean()))
    return {"logloss": ll, "brier": br, "accuracy": acc, "ece": float(ece)}


def apply_effects(base, events, effects):
    out = np.log(normalize(base))
    for i in range(len(out)):
        for spec in effects:
            if float(events.iloc[i].get(spec["key"], 0.0)) <= 0.0:
                continue
            c = int(spec["class_index"])
            if 0 <= c < out.shape[1]:
                out[i, c] += float(np.clip(spec["coefficient"], -MAX_ABS_LOGIT, MAX_ABS_LOGIT))
    z = out - np.max(out, axis=1, keepdims=True)
    e = np.exp(np.clip(z, -40.0, 40.0))
    return e / e.sum(axis=1, keepdims=True)


def fit_effects(df, base_cols, event_cols, n_classes):
    y = df["actual"].astype(int).to_numpy()
    base = normalize(df[base_cols].to_numpy(dtype=float))
    global_rate = np.bincount(y, minlength=n_classes).astype(float)
    global_rate = (global_rate + 1.0) / (len(y) + n_classes)
    effects = []
    for ev in event_cols:
        mask = pd.to_numeric(df[ev], errors="coerce").fillna(0).to_numpy() > 0.0
        n = int(mask.sum())
        if n < MIN_EVENT_ROWS:
            continue
        q = np.bincount(y[mask], minlength=n_classes).astype(float)
        q = (q + 1.0) / (n + n_classes)
        baseline_event = base[mask].mean(axis=0)
        for c in range(n_classes):
            q_c = float(np.clip(q[c], 1e-5, 1 - 1e-5))
            prior_c = float(np.clip(baseline_event[c] if np.isfinite(baseline_event[c]) else global_rate[c], 1e-5, 1 - 1e-5))
            raw = np.log(q_c / (1.0 - q_c)) - np.log(prior_c / (1.0 - prior_c))
            shrink = n / (n + SHRINKAGE)
            coef = float(np.clip(raw * shrink, -MAX_ABS_LOGIT, MAX_ABS_LOGIT))
            if abs(coef) > 0.005:
                effects.append({
                    "key": str(ev),
                    "class_index": int(c),
                    "coefficient": coef,
                    "max_abs_logit": MAX_ABS_LOGIT,
                    "support_rows": n,
                })
    return effects


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    candidates = sorted(CHECKPOINTS.glob("*_walkforward.csv"))
    if not candidates:
        OUTPUT.write_text(json.dumps({"status":"DEFERRED","reason":"No walk-forward checkpoints"}, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
        return 0

    records = []
    for path in candidates:
        df = pd.read_csv(path)
        if len(df) < MIN_ROWS or "actual" not in df.columns:
            continue
        if "baseline_context_free" not in df.columns:
            records.append({"checkpoint": str(path), "status":"DEFERRED","reason":"checkpoint is not explicitly marked baseline_context_free"})
            continue
        if not bool(pd.to_numeric(df["baseline_context_free"], errors="coerce").fillna(0).astype(bool).all()):
            records.append({"checkpoint": str(path), "status":"DEFERRED","reason":"baseline is not context-free; refusing potential double count"})
            continue
        base_cols = [c for c in ("pred_home", "pred_draw", "pred_away") if c in df.columns]
        n_classes = len(base_cols)
        if n_classes < 2:
            continue
        event_cols = sorted(c for c in df.columns if str(c).startswith("ctx_"))
        if not event_cols:
            records.append({"checkpoint": str(path), "status":"DEFERRED","reason":"No ctx_* columns"})
            continue
        if "datetime" in df.columns:
            df = df.sort_values(["datetime","game_id"], kind="mergesort").reset_index(drop=True)
        cut = max(80, int(len(df)*0.70))
        if len(df)-cut < 50:
            records.append({"checkpoint": str(path), "status":"DEFERRED","reason":"Validation segment too small"})
            continue

        train = df.iloc[:cut].copy()
        val = df.iloc[cut:].copy()
        effects = fit_effects(train, base_cols, event_cols, n_classes)
        if not effects:
            records.append({"checkpoint": str(path), "status":"DEFERRED","reason":"No supported event"})
            continue

        y_val = val["actual"].astype(int).to_numpy()
        base_val = normalize(val[base_cols].to_numpy(dtype=float))
        context_val = apply_effects(base_val, val[event_cols], effects)
        bm = metrics(base_val, y_val)
        cm = metrics(context_val, y_val)

        # Learn how strongly the context layer should be fused with the
        # context-free baseline. The validation segment chooses alpha only
        # after the event effects themselves have been frozen from train.
        best_alpha = 0.0
        best_mix_metrics = bm
        for alpha in np.linspace(0.0, 1.0, 21):
            mix = normalize((1.0-alpha) * base_val + alpha * context_val)
            mm = metrics(mix, y_val)
            if mm["logloss"] < best_mix_metrics["logloss"]:
                best_alpha = float(alpha)
                best_mix_metrics = mm

        delta = {k: float(best_mix_metrics[k]-bm[k]) for k in bm}
        candidate_ok = (
            best_alpha > 0.0
            and delta["logloss"] <= -MIN_LL
            and delta["brier"] <= -MIN_BRIER
            and delta["accuracy"] >= -MAX_ACC_REG
            and delta["ece"] <= MAX_ECE_REG
        )
        records.append({
            "checkpoint": str(path),
            "status": "PASS",
            "candidate_eligible": bool(candidate_ok),
            "fit_rows": int(len(train)),
            "validation_rows": int(len(val)),
            "effects": effects,
            "validation_baseline": bm,
            "validation_context": cm,
            "validation_fused": best_mix_metrics,
            "validation_delta": delta,
            "fusion_alpha": best_alpha,
        })

    passed = [r for r in records if r.get("status")=="PASS" and r.get("candidate_eligible")]
    payload = {
        "schema_version": 1,
        "artifact_type": "OOS_LEARNED_MATCHDAY_EFFECTS",
        "status": "PASS" if passed else "DEFERRED",
        "candidate_eligible": bool(passed),
        "effects": passed[0]["effects"] if passed else [],
        "fusion_alpha": passed[0]["fusion_alpha"] if passed else None,
        "results": records,
        "production_auto_promotion": False,
        "reason": None if passed else "No context candidate cleared the later chronological validation segment",
    }
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
