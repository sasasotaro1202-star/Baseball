"""Production Baseball research engine.

This module is the single research entry point for NPB + MLB. The existing
BaseballBacktest remains the numerical prediction core; this engine owns the
research lifecycle around it:

1. execute chronological OOS backtests;
2. persist auditable research state and weakness discovery;
3. keep Development OOS / Candidate Lock / Locked Holdout separate;
4. never promote a candidate from this baseline run alone.

Candidate promotion is delegated to ``research.validation_pipeline`` so the
independent holdout cannot be used during candidate selection.
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

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


@dataclass(frozen=True)
class ResearchCycle:
    """Auditable record for one production research execution."""

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
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def _cycle_id(started_at: str, commit: str) -> str:
    raw = f"{started_at}|{commit}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _research_state() -> dict[str, Any]:
    """Build weakness/research state from the verified OOS artifacts."""
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
            raise RuntimeError(
                "NPB results artifact is missing explicit three-way fields: "
                + ", ".join(sorted(missing))
            )
        for row in reader:
            p = validate_npb_probabilities([
                float(row["pred_home"]),
                float(row["pred_draw"]),
                float(row["pred_away"]),
            ])
            rows += 1
            h = float(row["actual_home_score"])
            a = float(row["actual_away_score"])
            if h == a:
                draw_actual += 1
            if int(max(range(3), key=lambda i: p[i])) == 1:
                draw_predicted += 1

    contract = {
        "labels": list(NPB_OUTCOME_LABELS),
        "rows_verified": rows,
        "actual_draw_rows": draw_actual,
        "predicted_draw_rows": draw_predicted,
        "draw_field_required": True,
        "probability_order": ["pred_home", "pred_draw", "pred_away"],
        "status": "PASS",
    }
    _write_json(ROOT / "results" / "npb_outcome_contract.json", contract)
    return contract


class BaseballResearchEngine:
    """Production orchestrator around the existing NPB/MLB backtest core."""

    ENGINE_VERSION = "baseball-research-engine-v1"

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
            "decision": "ADOPT only when research.validation_pipeline permits it",
        }
        _write_json(ROOT / "research_state.json", state)
        return state

    def run(
        self,
        *,
        npb: bool = True,
        mlb: bool = True,
        mlb_start: int = 2020,
        mlb_end: int = 2026,
    ) -> ResearchCycle:
        RESULTS.mkdir(parents=True, exist_ok=True)
        self._backtest(npb=npb, mlb=mlb, mlb_start=mlb_start, mlb_end=mlb_end)
        self._research_state()
        self.stages.append("candidate_selection_not_implicit")
        self.stages.append("independent_holdout_required_for_promotion")

        finished = datetime.now(timezone.utc).isoformat()
        cycle = ResearchCycle(
            cycle_id=self.cycle_id,
            git_commit=self.git_commit,
            started_at=self.started_at,
            finished_at=finished,
            leagues=tuple(x for x, enabled in (("NPB", npb), ("MLB", mlb)) if enabled),
            stages=tuple(self.stages),
            promotion_decision="NO_CHANGE",
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
