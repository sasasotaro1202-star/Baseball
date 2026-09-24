"""Research-only chronological replay for drift/uncertainty-aware expert routing.

Consumes walk-forward checkpoints containing expert_* probability columns produced
by baseball_backtest.py. It never refits models and never uses the current outcome
until after the current prediction has been scored. Missing checkpoints are a
clean DEFERRED state rather than an error.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from research.drift_uncertainty_routing import (
    AdaptiveTemperatureCalibrator,
    ExpertCalibrationBank,
    RoutingConfig,
    mix_expert_probabilities,
    route_experts,
)


RESULTS = Path("results")
CHECKPOINTS = Path("data/checkpoints")


def _ece(probs: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    p = np.asarray(probs, dtype=float)
    labels = np.asarray(y, dtype=int)
    if len(p) == 0:
        return float("nan")
    conf = p.max(axis=1)
    pred = p.argmax(axis=1)
    err = (pred != labels).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf < hi if hi < 1.0 else conf <= hi)
        if np.any(m):
            total += float(m.mean()) * abs(float(1.0 - err[m].mean()) - float(conf[m].mean()))
    return float(total)


def _metrics(probs: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    p = np.asarray(probs, dtype=float)
    labels = np.asarray(y, dtype=int)
    if len(p) != len(labels) or len(p) == 0:
        raise ValueError("probabilities/labels mismatch")
    p = np.clip(p, 1e-12, 1.0)
    p /= p.sum(axis=1, keepdims=True)
    ll = float(np.mean(-np.log(p[np.arange(len(labels)), labels])))
    target = np.zeros_like(p)
    target[np.arange(len(labels)), labels] = 1.0
    brier = float(np.mean(np.sum((p - target) ** 2, axis=1)))
    acc = float(np.mean(p.argmax(axis=1) == labels))
    return {"accuracy": acc, "logloss": ll, "brier": brier, "ece": _ece(p, labels)}


def _expert_matrix(row: pd.Series, keys: List[str], league: str) -> np.ndarray:
    cols = []
    for key in keys:
        names = [f"expert_{key}_home"]
        if league == "NPB":
            names += [f"expert_{key}_draw", f"expert_{key}_away"]
        else:
            names += [f"expert_{key}_away"]
        if not all(n in row.index for n in names):
            raise ValueError(f"missing expert columns for {key}")
        values = [float(row[n]) for n in names]
        cols.append(values)
    p = np.asarray(cols, dtype=float)
    if not np.all(np.isfinite(p)):
        raise ValueError("non-finite expert probability in checkpoint")
    p = np.clip(p, 1e-12, 1.0)
    p /= p.sum(axis=1, keepdims=True)
    return p


def _predictive_drift(history: List[np.ndarray], current: np.ndarray) -> float:
    """Output-space drift proxy using only earlier expert predictions.

    recent mean is compared with an older mean. This is target-free and exists
    even when historical raw feature vectors were not persisted.
    """

    if len(history) < 24:
        return 0.0
    arr = np.asarray(history, dtype=float)
    short = arr[-12:].mean(axis=0)
    old = arr[max(0, len(arr) - 96):-12]
    if len(old) < 12:
        return 0.0
    baseline = old.mean(axis=0)
    diff = float(np.mean(np.abs(short - baseline)))
    current_shift = float(np.mean(np.abs(current - short)))
    raw = 0.75 * diff + 0.25 * current_shift
    return float(np.clip(1.0 - np.exp(-raw / 0.12), 0.0, 1.0))


def _safe_numeric_pregame_columns(df: pd.DataFrame) -> List[str]:
    """Allowlist only explicitly pregame feature columns for drift detection."""
    allowed = []
    for c in df.columns:
        name = str(c).lower()
        if not (name.startswith("feature_") or name.startswith("pregame_")):
            continue
        if name in {"feature_actual", "feature_target", "pregame_actual", "pregame_target"}:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            allowed.append(c)
    return sorted(allowed)


def _feature_drift(history: List[np.ndarray], current: np.ndarray) -> float:
    if len(history) < 24:
        return 0.0
    arr = np.asarray(history, dtype=float)
    ref = arr[max(0, len(arr) - 96):-12]
    recent = arr[-12:]
    cur = np.asarray(current, dtype=float).reshape(1, -1)
    if len(ref) < 12:
        return 0.0
    try:
        from research.drift_uncertainty_routing import feature_drift_score, mmd_drift_score
        loc_scale = feature_drift_score(ref, recent)
        # MMD can compare different sample sizes, so the current pregame row
        # is included here without using any target/postgame field.
        mmd = mmd_drift_score(recent, cur)
        return float(np.clip(0.50 * loc_scale + 0.50 * mmd, 0.0, 1.0))
    except Exception:
        return 0.0


def _replay(df: pd.DataFrame, config: RoutingConfig) -> Tuple[Dict, Dict]:
    league = str(df["league"].iloc[0])
    expert_cols = sorted({c[len("expert_"):].rsplit("_", 1)[0] for c in df.columns if c.startswith("expert_") and c.endswith("_home")})
    if len(expert_cols) < 2:
        raise ValueError("fewer than two expert probability columns")

    if "datetime" in df.columns:
        df = df.sort_values(["datetime", "game_id"], kind="mergesort").reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    base_probs = []
    routed_probs = []
    mixed_probs = []
    outcomes = []
    history_logloss = []
    history_expert_probs = []
    raw_expert_history = []
    expert_calibrators = ExpertCalibrationBank(len(expert_cols), config=config)
    feature_cols = _safe_numeric_pregame_columns(df)
    history_pregame_features = []
    previous_weights = None
    calibrator = AdaptiveTemperatureCalibrator(config=config)
    routing_rows = []

    for i, row in df.iterrows():
        current_raw = _expert_matrix(row, expert_cols, league)
        current = expert_calibrators.predict(current_raw)

        # Critical chronology rule: history arrays contain only already-resolved rows.
        if history_logloss:
            hist = np.asarray(history_logloss[-max(config.long_window, 120):], dtype=float)
        else:
            hist = np.empty((0, len(expert_cols)), dtype=float)
        output_drift = _predictive_drift(history_expert_probs, current)
        current_features = None
        feature_drift = 0.0
        feature_mmd_drift = 0.0
        if feature_cols:
            values = pd.to_numeric(row[feature_cols], errors="coerce").to_numpy(dtype=float)
            if np.all(np.isfinite(values)):
                current_features = values
                feature_drift = _feature_drift(history_pregame_features, current_features)
                if len(history_pregame_features) >= 24:
                    try:
                        from research.drift_uncertainty_routing import mmd_drift_score
                        arr = np.asarray(history_pregame_features, dtype=float)
                        ref = arr[max(0, len(arr) - 96):-12]
                        recent = arr[-12:]
                        feature_mmd_drift = float(
                            mmd_drift_score(recent, current_features.reshape(1, -1))
                        )
                    except Exception:
                        feature_mmd_drift = 0.0
        if feature_cols:
            drift = float(np.clip(
                0.55 * output_drift + 0.225 * feature_drift + 0.225 * feature_mmd_drift,
                0.0, 1.0
            ))
        else:
            drift = output_drift

        if len(hist) >= 8:
            rr = route_experts(
                expert_probs=current,
                history_logloss=hist,
                drift_score=drift,
                previous_weights=previous_weights,
                config=config,
            )
        else:
            k = len(expert_cols)
            rr = type("BootstrapRouting", (), {
                "weights": np.full(k, 1.0 / k),
                "short_loss": np.full(k, np.nan),
                "long_loss": np.full(k, np.nan),
                "drift_score": drift,
                "disagreement": float(np.mean(np.std(current, axis=0))),
                "predictive_entropy": 0.0,
                "uncertainty": 0.0,
            })()
        mixed = mix_expert_probabilities(current, rr.weights)
        final = calibrator.predict(mixed)

        # Outcome enters only after final prediction is produced.
        y = int(row["actual"])
        base = [float(row.get("pred_home", np.nan))]
        if league == "NPB":
            base += [float(row.get("pred_draw", np.nan)), float(row.get("pred_away", np.nan))]
        else:
            base += [float(row.get("pred_away", np.nan))]
        if not all(np.isfinite(base)):
            raise ValueError("baseline prediction columns are missing/non-finite")

        base_probs.append(base)
        mixed_probs.append(mixed.tolist())
        routed_probs.append(final.tolist())
        outcomes.append(y)
        routing_rows.append({
            "game_id": str(row["game_id"]),
            "drift_score": drift,
            "output_drift_score": output_drift,
            "feature_drift_score": feature_drift,
            "feature_mmd_drift_score": feature_mmd_drift,
            "uncertainty": rr.uncertainty,
            "disagreement": rr.disagreement,
            "temperature_before_update": calibrator.temperature,
            "weights": rr.weights.tolist(),
        })

        # Predict-then-update: current outcome becomes eligible for future routing.
        row_losses = []
        for ep in current:
            row_losses.append(float(-np.log(max(float(ep[y]), 1e-12))))
        history_logloss.append(row_losses)
        history_expert_probs.append(current)
        raw_expert_history.append(current_raw)
        if current_features is not None:
            history_pregame_features.append(current_features)
        previous_weights = rr.weights.copy()
        # Both expert-level and final calibration obey predict-then-update.
        if len(raw_expert_history) >= 2:
            expert_calibrators.update(
                np.asarray(raw_expert_history[-config.long_window:]),
                np.asarray(outcomes[-config.long_window:]),
            )
        calibrator.update(np.asarray(mixed_probs[-config.long_window:]), np.asarray(outcomes[-config.long_window:]))

    base_arr = np.asarray(base_probs, dtype=float)
    routed_arr = np.asarray(routed_probs, dtype=float)
    y_arr = np.asarray(outcomes)
    baseline_metrics = _metrics(base_arr, y_arr)
    routed_metrics = _metrics(routed_arr, y_arr)
    delta = {f"delta_{k}": float(routed_metrics[k] - baseline_metrics[k]) for k in baseline_metrics}

    diag_df = pd.DataFrame(routing_rows)
    diagnostics = {
        "calibration_updates_accepted": int(calibrator.accepted_updates),
        "calibration_updates_rejected": int(calibrator.rejected_updates),
        "temperature_start": 1.0,
        "temperature_end": float(calibrator.temperature),
        "expert_calibration_updates_accepted": int(expert_calibrators.accepted_updates),
        "expert_calibration_updates_rejected": int(expert_calibrators.rejected_updates),
        "expert_temperatures_end": [float(x) for x in expert_calibrators.temperatures()],
    }
    if not diag_df.empty:
        for col in ("drift_score", "uncertainty"):
            if col in diag_df.columns:
                q = float(diag_df[col].median())
                mask = diag_df[col].to_numpy(dtype=float) >= q
                if int(mask.sum()) >= 20:
                    sub_base = base_arr[mask]
                    sub_route = routed_arr[mask]
                    sub_y = y_arr[mask]
                    diagnostics[f"high_{col}_rows"] = int(mask.sum())
                    diagnostics[f"high_{col}_delta_logloss"] = float(
                        _metrics(sub_route, sub_y)["logloss"] - _metrics(sub_base, sub_y)["logloss"]
                    )

    windows = {}
    n_rows = len(y_arr)
    if n_rows >= 200:
        half = max(50, n_rows // 4)
        starts = [max(0, n_rows - 2 * half), max(0, n_rows - half)]
        for idx, start in enumerate(starts, 1):
            end = min(n_rows, start + half)
            if end - start < 50:
                continue
            bm = _metrics(base_arr[start:end], y_arr[start:end])
            rm = _metrics(routed_arr[start:end], y_arr[start:end])
            windows[f"window_{idx}"] = {
                "start_index": int(start),
                "end_index": int(end),
                "rows": int(end - start),
                "baseline": bm,
                "routed_recalibrated": rm,
                "delta": {f"delta_{k}": float(rm[k] - bm[k]) for k in bm},
            }

    artifact = {
        "status": "PASS",
        "league": league,
        "rows": int(len(df)),
        "experts": expert_cols,
        "feature_drift_columns": feature_cols,
        "baseline": baseline_metrics,
        "routed_recalibrated": routed_metrics,
        "delta": delta,
        "late_oos_windows": windows,
        "final_temperature": float(calibrator.temperature),
        "diagnostics": diagnostics,
        "routing_rows_tail": routing_rows[-20:],
    }
    return artifact, delta


def main() -> int:
    candidates = sorted(CHECKPOINTS.glob("*_walkforward.csv"))
    candidates = [p for p in candidates if p.is_file()]
    if not candidates:
        payload = {
            "status": "DEFERRED",
            "reason": "No walk-forward checkpoint with expert probabilities is available yet.",
            "checkpoint_dir": str(CHECKPOINTS),
        }
        RESULTS.mkdir(parents=True, exist_ok=True)
        (RESULTS / "routing_oos_replay.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return 0

    config = RoutingConfig()
    artifacts = []
    for path in candidates:
        try:
            df = pd.read_csv(path)
            if len(df) < 50 or "actual" not in df.columns or "league" not in df.columns:
                continue
            art, _ = _replay(df, config)
            art["checkpoint"] = str(path)
            artifacts.append(art)
        except Exception as exc:
            # A single stale/legacy checkpoint must not be mistaken for success.
            artifacts.append({
                "status": "DEFERRED",
                "checkpoint": str(path),
                "reason": f"checkpoint not replayable: {exc}",
            })

    payload = {"schema_version": 1, "results": artifacts}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "routing_oos_replay.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
