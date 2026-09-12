"""Leakage-safe Development OOS candidate generation and locking.

This module deliberately stops before independent holdout evaluation.  It turns
verified Development OOS artifacts into a deterministic candidate plan and a
Candidate Lock record.  No holdout file is read here.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    league: str
    objective: str
    model_version: str
    feature_version: str
    development_metrics: dict[str, float]
    selection_reason: str
    git_commit: str
    dataset_hash: str


def _finite(value: Any, default: float | None = None) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if value == value and abs(value) != float("inf") else default


def _read_csv(path: Path) -> list[dict[str, Any]]:
    import csv

    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _artifact_hash(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for path in sorted(paths, key=lambda p: str(p)):
        if not path.exists():
            continue
        h.update(str(path.relative_to(ROOT)).encode("utf-8"))
        h.update(path.read_bytes())
    return h.hexdigest()


def _candidate_id(*, league: str, objective: str, model_version: str,
                  feature_version: str, git_commit: str, dataset_hash: str) -> str:
    raw = "|".join((league, objective, model_version, feature_version, git_commit, dataset_hash))
    return "cand-" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def _metrics(row: Mapping[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in ("Accuracy", "LogLoss", "Brier", "ECE", "ScoreMAE", "LowHighAccuracy", "rows"):
        value = _finite(row.get(key))
        if value is not None:
            out[key] = value
    return out


def select_development_candidate(
    *,
    league: str,
    objective: str,
    git_commit: str,
    feature_version: str,
    baseline_model: str | None = None,
    min_rows: int = 200,
) -> CandidateSpec | None:
    """Select exactly one candidate using Development OOS only.

    The source is the league model-comparison artifact produced by the existing
    chronological backtest.  The function fails closed when the artifact does
    not identify enough information to distinguish a candidate from baseline.
    No holdout artifact is opened or inspected.
    """
    path = RESULTS / f"{league.lower()}_model_comparison.csv"
    rows = _read_csv(path)
    if not rows:
        return None

    usable: list[tuple[dict[str, Any], dict[str, float]]] = []
    for row in rows:
        metrics = _metrics(row)
        if metrics.get("rows", 0) < min_rows:
            continue
        model = str(row.get("model", row.get("Model", ""))).strip()
        if not model or "LogLoss" not in metrics or "Brier" not in metrics or "Accuracy" not in metrics:
            continue
        usable.append((row, metrics))
    if not usable:
        return None

    baseline = (baseline_model or "").strip()
    if not baseline:
        # Never infer the production baseline from the candidate leaderboard.
        # It must be explicitly declared by the production manifest/environment.
        return None

    alternatives = [(row, metrics) for row, metrics in usable
                    if str(row.get("model", row.get("Model", ""))).strip() != baseline]
    if not alternatives:
        return None

    alternatives.sort(key=lambda item: (
        item[1].get("LogLoss", float("inf")),
        item[1].get("Brier", float("inf")),
        -item[1].get("Accuracy", 0.0),
        str(item[0].get("model", item[0].get("Model", ""))),
    ))
    row, metrics = alternatives[0]
    model_version = str(row.get("model", row.get("Model", ""))).strip()
    dataset_hash = _artifact_hash([path, RESULTS / f"{league.lower()}_backtest_results.csv"])
    candidate_id = _candidate_id(
        league=league,
        objective=objective,
        model_version=model_version,
        feature_version=feature_version,
        git_commit=git_commit,
        dataset_hash=dataset_hash,
    )
    return CandidateSpec(
        candidate_id=candidate_id,
        league=league,
        objective=objective,
        model_version=model_version,
        feature_version=feature_version,
        development_metrics=metrics,
        selection_reason=(
            "Selected on Development OOS only: lowest LogLoss, then Brier, "
            "then highest Accuracy among models different from the declared baseline."
        ),
        git_commit=git_commit,
        dataset_hash=dataset_hash,
    )


def lock_candidate(spec: CandidateSpec) -> dict[str, Any]:
    """Persist an immutable Candidate Lock artifact; never evaluates holdout."""
    locked_at = datetime.now(timezone.utc).isoformat()
    payload = {
        "schema_version": 1,
        "stage": "candidate_locked",
        "locked_at": locked_at,
        "candidate": asdict(spec),
        "holdout_evaluated": False,
        "holdout_access": "forbidden_during_selection",
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    _write = RESULTS / f"{spec.league.lower()}_candidate_lock.json"
    _write.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return payload
