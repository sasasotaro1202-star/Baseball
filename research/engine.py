"""Production Baseball research engine.

The numerical prediction core remains ``BaseballBacktest``.  This module owns
an auditable research lifecycle around it:

chronological OOS -> weakness discovery -> Development candidate selection ->
Candidate Lock -> independent Locked Holdout -> ADOPT/REJECT/HOLD.

The NPB production target is explicitly Home / Draw / Away.  No candidate is
promoted when the independent holdout artifact is missing or incomplete.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from baseball_backtest import BaseballBacktest
from evaluation.npb_outcome import NPB_OUTCOME_LABELS, validate_npb_probabilities
from research.candidates import lock_candidate, select_development_candidate
from research.candidate_registry import record_candidate

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


@dataclass(frozen=True)
class ResearchCycle:
    cycle_id: str
    git_commit: str
    started_at: str
    finished_at: str | None
    leagues: tuple[str, ...]
    stages: tuple[str, ...]
    promotion_decision: str


def _git_commit() -> str:
    value = os.getenv("GITHUB_SHA", "").strip()
    if value:
        return value
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def _cycle_id(started_at: str, commit: str) -> str:
    return hashlib.sha256(f"{started_at}|{commit}".encode("utf-8")).hexdigest()[:16]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _research_state() -> dict[str, Any]:
    from research.research_loop import build_state
    return build_state()


def _verify_npb_outcome_contract() -> dict[str, Any]:
    """Fail closed if an NPB OOS artifact has collapsed Draw into binary form."""
    path = RESULTS / "npb_backtest_results.csv"
    if not path.exists():
        raise RuntimeError("NPB results artifact is missing; cannot verify Draw output")
    rows = 0
    draw_actual = 0
    draw_predicted = 0
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        required = {"pred_home", "pred_draw", "pred_away", "actual_home_score", "actual_away_score"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise RuntimeError("NPB results artifact is missing explicit three-way fields: " + ", ".join(sorted(missing)))
        for row in reader:
            p = validate_npb_probabilities([float(row["pred_home"]), float(row["pred_draw"]), float(row["pred_away"])])
            rows += 1
            h = float(row["actual_home_score"]); a = float(row["actual_away_score"])
            draw_actual += int(h == a)
            draw_predicted += int(int(max(range(3), key=lambda i: p[i])) == 1)
    contract = {
        "labels": list(NPB_OUTCOME_LABELS),
        "rows_verified": rows,
        "actual_draw_rows": draw_actual,
        "predicted_draw_rows": draw_predicted,
        "draw_field_required": True,
        "probability_order": ["pred_home", "pred_draw", "pred_away"],
        "status": "PASS",
    }
    _write_json(RESULTS / "npb_outcome_contract.json", contract)
    return contract


def _load_baseline_model() -> str | None:
    """Production baseline must be explicitly declared; never infer it from candidates."""
    env = os.getenv("BASEBALL_NPB_BASELINE_MODEL", "").strip()
    if env:
        return env
    manifest = RESULTS / "npb_production_manifest.json"
    if manifest.exists():
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            value = str(payload.get("model_version", "")).strip()
            return value or None
        except Exception:
            return None
    return None


def _load_locked_holdout(candidate_id: str) -> dict[str, Any] | None:
    """Read only the already-created independent holdout artifact after Candidate Lock."""
    path = RESULTS / "npb_locked_holdout.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"invalid independent NPB holdout artifact: {exc}") from exc
    if payload.get("stage") != "locked_holdout_ready":
        raise RuntimeError("NPB holdout artifact is not marked locked_holdout_ready")
    if payload.get("candidate_id") != candidate_id:
        raise RuntimeError("NPB holdout candidate_id does not match Candidate Lock")
    required = {
        "baseline", "candidate", "validation_windows", "calibration_ok",
        "no_future_target_data", "reproducible", "baseline_score",
        "candidate_score", "baseline_hilo", "candidate_hilo",
    }
    missing = required - set(payload)
    if missing:
        raise RuntimeError("NPB holdout artifact missing required fields: " + ", ".join(sorted(missing)))
    return payload


def _npb_research_lifecycle(git_commit: str) -> dict[str, Any]:
    """Run candidate selection/lock/holdout decision without ever fabricating evidence."""
    baseline_model = _load_baseline_model()
    state = _research_state()
    focus = state.get("focus", {})
    objective = str(focus.get("objective", "win"))
    feature_version = os.getenv("BASEBALL_NPB_FEATURE_VERSION", "baseball-features-v1")

    if not baseline_model:
        return {
            "stage": "baseline_declaration_required",
            "decision": "HOLD",
            "reason": "NPB production baseline model is not explicitly declared",
        }

    spec = select_development_candidate(
        league="NPB",
        objective=objective,
        git_commit=git_commit,
        feature_version=feature_version,
        baseline_model=baseline_model,
    )
    if spec is None:
        return {
            "stage": "candidate_selection",
            "decision": "NO_CHANGE",
            "reason": "No reproducible Development OOS candidate was eligible against the declared baseline",
            "baseline_model": baseline_model,
        }

    locked = lock_candidate(spec)
    holdout = _load_locked_holdout(spec.candidate_id)
    if holdout is None:
        return {
            "stage": "candidate_locked",
            "decision": "HOLD",
            "reason": "Independent locked holdout artifact is not available",
            "candidate": locked,
        }

    record = record_candidate(
        candidate_id=spec.candidate_id,
        git_commit=git_commit,
        feature_version=spec.feature_version,
        model_version=spec.model_version,
        development_metrics=spec.development_metrics,
        holdout_baseline=holdout["baseline"],
        holdout_candidate=holdout["candidate"],
        validation_windows=int(holdout["validation_windows"]),
        calibration_ok=bool(holdout["calibration_ok"]),
        no_future_target_data=bool(holdout["no_future_target_data"]),
        reproducible=bool(holdout["reproducible"]),
        holdout_score_baseline=holdout["baseline_score"],
        holdout_score_candidate=holdout["candidate_score"],
        holdout_hilo_baseline=holdout["baseline_hilo"],
        holdout_hilo_candidate=holdout["candidate_hilo"],
        league="NPB",
    )
    return {
        "stage": "locked_holdout_evaluated",
        "decision": record.decision,
        "candidate_id": record.candidate_id,
        "candidate_model": record.model_version,
        "baseline_model": baseline_model,
        "registry": str(RESULTS / "candidate_registry.json"),
        "record": asdict(record),
    }


class BaseballResearchEngine:
    ENGINE_VERSION = "baseball-research-engine-v2-npb-lifecycle"

    def __init__(self, data_dir: str | Path = "data") -> None:
        self.data_dir = Path(data_dir)
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.git_commit = _git_commit()
        self.cycle_id = _cycle_id(self.started_at, self.git_commit)
        self.stages: list[str] = []

    def _backtest(self, *, npb: bool, mlb: bool, mlb_start: int, mlb_end: int) -> None:
        self.stages.append("chronological_oos_backtest")
        bt = BaseballBacktest(self.data_dir)
        bt.run(npb=npb, mlb=mlb, mlb_start=mlb_start, mlb_end=mlb_end)
        if npb:
            self.stages.append("npb_home_draw_away_contract_check")
            _verify_npb_outcome_contract()

    def _research_state(self) -> dict[str, Any]:
        self.stages.append("weakness_discovery")
        state = _research_state()
        state["engine_version"] = self.ENGINE_VERSION
        state["cycle_id"] = self.cycle_id
        state["git_commit"] = self.git_commit
        state["promotion_policy"] = {
            "development_oos": "candidate selection only",
            "candidate_lock": "required before holdout",
            "locked_holdout": "independent confirmation only",
            "npb_target": "HOME / DRAW / AWAY plus DrawRecall and DrawProbabilityMAE",
            "decision": "ADOPT only when research.validation_pipeline permits it",
        }
        _write_json(ROOT / "research_state.json", state)
        return state

    def run(self, *, npb: bool = True, mlb: bool = True, mlb_start: int = 2020, mlb_end: int = 2026) -> ResearchCycle:
        RESULTS.mkdir(parents=True, exist_ok=True)
        self._backtest(npb=npb, mlb=mlb, mlb_start=mlb_start, mlb_end=mlb_end)
        self._research_state()
        promotion_decision = "NO_CHANGE"
        if npb:
            self.stages.append("npb_development_candidate_selection")
            lifecycle = _npb_research_lifecycle(self.git_commit)
            _write_json(RESULTS / "npb_research_lifecycle.json", lifecycle)
            self.stages.append(lifecycle["stage"])
            promotion_decision = lifecycle["decision"]
        else:
            self.stages.append("npb_lifecycle_skipped")

        finished = datetime.now(timezone.utc).isoformat()
        cycle = ResearchCycle(
            cycle_id=self.cycle_id,
            git_commit=self.git_commit,
            started_at=self.started_at,
            finished_at=finished,
            leagues=tuple(x for x, enabled in (("NPB", npb), ("MLB", mlb)) if enabled),
            stages=tuple(self.stages),
            promotion_decision=promotion_decision,
        )
        _write_json(ROOT / "research_cycle_manifest.json", asdict(cycle))
        return cycle


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m research.engine")
    parser.add_argument("--npb-only", action="store_true")
    parser.add_argument("--mlb-only", action="store_true")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--mlb-start", type=int, default=2020)
    parser.add_argument("--mlb-end", type=int, default=2026)
    args = parser.parse_args(argv)
    if args.npb_only and args.mlb_only:
        parser.error("--npb-only and --mlb-only are mutually exclusive")
    engine = BaseballResearchEngine(args.data_dir)
    cycle = engine.run(
        npb=not args.mlb_only,
        mlb=not args.npb_only,
        mlb_start=args.mlb_start,
        mlb_end=args.mlb_end,
    )
    print(json.dumps(asdict(cycle), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
