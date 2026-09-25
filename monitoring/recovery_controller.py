#!/usr/bin/env python3
from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

API = os.environ.get("GITHUB_API_URL", "https://api.github.com").rstrip("/")
REPO = os.environ["GITHUB_REPOSITORY"]
TOKEN = os.environ["GH_TOKEN"]

WORKFLOWS = {
    "validation": ["validate-code.yml"],
    "production": ["baseball_production.yml"],
    "acquisition": [
        "baseball-parallel-source-acquisition.yml",
        "baseball-data-acquisition.yml",
    ],
    "mlb_competition": ["baseball-mlb-competition.yml"],
    "mac": ["baseball_mac_compute.yml"],
    "research": ["baseball_research.yml"],
    "matchday": ["baseball_matchday.yml"],
    "game_type_ablation": ["baseball_game_type_ablation.yml"],
}

STALE_MINUTES = {
    "validation": 25,
    "production": 155,
    "acquisition": 165,
    "mlb_competition": 165,
    "mac": 180,
    "research": 180,
    "matchday": 45,
    "game_type_ablation": 180,
}
QUEUED_STALE_MINUTES = 20
ACTIVE_STATES = {"queued", "in_progress", "waiting", "pending", "requested"}
QUEUED_STATES = {"queued", "waiting", "pending", "requested"}
RECENT_DISPATCH_GUARD_MINUTES = 5
RETRYABLE = {408, 409, 425, 429, 500, 502, 503, 504}


def request(method: str, path: str, payload: dict | None = None, attempts: int = 5):
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    last = None
    for attempt in range(1, attempts + 1):
        req = urllib.request.Request(
            API + path,
            data=body,
            method=method,
            headers={
                "Authorization": f"Bearer {TOKEN}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "Baseball-Autonomous-Recovery",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = resp.read()
                return json.loads(raw.decode("utf-8")) if raw else {}
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            last = RuntimeError(
                f"GitHub API {method} {path} failed: HTTP {exc.code}: {detail[:1200]}"
            )
            if exc.code not in RETRYABLE:
                raise last
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last = exc
        if attempt < attempts:
            time.sleep(min(20, 2 ** (attempt - 1)))
    raise last


def ts(row: dict | None) -> datetime:
    value = (row or {}).get("created_at") or (row or {}).get("updated_at")
    if not value:
        return datetime.min.replace(tzinfo=timezone.utc)
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def active(row: dict | None) -> bool:
    return bool(row) and row.get("status") in ACTIVE_STATES


def success(row: dict | None) -> bool:
    return bool(row) and row.get("conclusion") == "success"


def age_minutes(row: dict | None, now: datetime) -> float:
    return (now - ts(row)).total_seconds() / 60.0 if row else 0.0


def recently_created(rows: list[dict], now: datetime) -> bool:
    return any(0.0 <= age_minutes(row, now) < RECENT_DISPATCH_GUARD_MINUTES for row in rows)


def runs(filename: str) -> list[dict]:
    return request(
        "GET",
        f"/repos/{REPO}/actions/workflows/{filename}/runs?per_page=100",
    ).get("workflow_runs", [])


def main_sha() -> str:
    payload = request("GET", f"/repos/{REPO}/git/ref/heads/main")
    return str(payload.get("object", {}).get("sha") or "")


def readiness(path: str) -> tuple[bool, int]:
    try:
        payload = request(
            "GET", f"/repos/{REPO}/contents/{path}?ref=main"
        )
        raw = base64.b64decode(payload.get("content", "")).decode("utf-8")
        state = json.loads(raw)
        return bool(state.get("complete", False)), int(state.get("aggregate_games") or 0)
    except Exception as exc:
        print(f"[WATCHDOG] readiness lookup deferred for {path}: {exc}")
        return False, 0


def main() -> int:
    now = datetime.now(timezone.utc)
    errors: list[str] = []
    stuck: list[int] = []
    dispatched: list[str] = []

    data: dict[str, list[dict]] = {}
    logical_latest: dict[str, dict | None] = {}
    logical_active: dict[str, list[dict]] = {}

    for label, filenames in WORKFLOWS.items():
        combined: list[dict] = []
        for filename in filenames:
            for row in runs(filename):
                row = dict(row)
                row["_workflow_file"] = filename
                combined.append(row)
        combined.sort(key=ts, reverse=True)
        data[label] = combined
        logical_latest[label] = combined[0] if combined else None
        logical_active[label] = [row for row in combined if active(row)]

    for label, rows in logical_active.items():
        parallel_acquisition_active = (
            label == "acquisition"
            and any(
                row.get("_workflow_file") == "baseball-parallel-source-acquisition.yml"
                for row in rows
            )
        )
        for row in rows:
            legacy_acquisition_superseded = (
                label == "acquisition"
                and row.get("_workflow_file") == "baseball-data-acquisition.yml"
                and parallel_acquisition_active
            )
            limit = (
                0.0
                if legacy_acquisition_superseded
                else (
                    QUEUED_STALE_MINUTES
                    if row.get("status") in QUEUED_STATES
                    else STALE_MINUTES[label]
                )
            )
            if age_minutes(row, now) <= limit:
                continue
            run_id = int(row["id"])
            print(
                f"[WATCHDOG] STUCK {label} run={run_id} status={row.get('status')} "
                f"age_min={age_minutes(row, now):.1f} limit={limit}; cancelling"
            )
            try:
                request("POST", f"/repos/{REPO}/actions/runs/{run_id}/cancel")
                stuck.append(run_id)
            except Exception as exc:
                errors.append(f"cancel {run_id}: {exc}")

    # A production OOS run is immutable to its evaluated SHA. When main
    # advances, any older active production run cannot safely persist its result
    # and only consumes a runner. Cancel it early; its checkpointed work can be
    # resumed under the current validated SHA.
    for row in logical_active.get("production", []):
        row_sha = str(row.get("head_sha") or "")
        if row_sha and current_sha and row_sha != current_sha:
            run_id = int(row["id"])
            print(
                f"[WATCHDOG] STALE production run={run_id} head_sha={row_sha} "
                f"current_main={current_sha}; cancelling"
            )
            try:
                request("POST", f"/repos/{REPO}/actions/runs/{run_id}/cancel")
                stuck.append(run_id)
            except Exception as exc:
                errors.append(f"cancel stale production {run_id}: {exc}")

    npb_ready, npb_games = readiness("data/checkpoints/npb_collection_status.json")
    mlb_ready, mlb_games = readiness("data/checkpoints/mlb_collection_status.json")
    validation = logical_latest["validation"]
    production = logical_latest["production"]
    acquisition = logical_latest["acquisition"]
    mlb_competition = logical_latest["mlb_competition"]
    mac = logical_latest["mac"]
    research = logical_latest["research"]
    matchday = logical_latest["matchday"]
    game_type_ablation = logical_latest["game_type_ablation"]

    current_sha = main_sha()
    validation_current = success(validation) and str(validation.get("head_sha") or "") == current_sha

    print(f"[WATCHDOG] main_sha={current_sha}")
    print(f"[WATCHDOG] validation_current={validation_current}")
    print(f"[WATCHDOG] NPB ready={npb_ready} games={npb_games}")
    print(f"[WATCHDOG] MLB ready={mlb_ready} games={mlb_games}")

    def any_active(label: str) -> bool:
        return bool(logical_active.get(label))

    def active_current_sha(label: str, sha: str) -> bool:
        return any(
            active(row) and str(row.get("head_sha") or "") == str(sha or "")
            for row in logical_active.get(label, [])
        )

    def dispatch(label: str, workflow_file: str) -> None:
        try:
            request(
                "POST",
                f"/repos/{REPO}/actions/workflows/{workflow_file}/dispatches",
                {"ref": "main"},
            )
            dispatched.append(label)
            print(f"[WATCHDOG] dispatched {label}: {workflow_file}")
        except Exception as exc:
            errors.append(f"dispatch {workflow_file}: {exc}")
            print(f"[WATCHDOG] dispatch failed for {workflow_file}: {exc}")

    # A validation run on an older SHA must not block validation of current main.
    # The validation workflow has cancel-in-progress enabled, so dispatching the
    # current SHA safely supersedes the stale active run.
    if not validation_current and not active_current_sha("validation", current_sha) and not recently_created(
        [r for r in data["validation"] if str(r.get("head_sha") or "") == str(current_sha or "")], now
    ):
        dispatch("validation", "validate-code.yml")

    if not any_active("matchday") and not recently_created(data["matchday"], now):
        dispatch("matchday", "baseball_matchday.yml")

    mlb_competition_failed = (
        mlb_competition is None
        or mlb_competition.get("conclusion") in {"failure", "cancelled", "timed_out"}
        or age_minutes(mlb_competition, now) > STALE_MINUTES["mlb_competition"]
    )
    if (
        mlb_competition_failed
        and not any_active("mlb_competition")
        and not recently_created(data["mlb_competition"], now)
    ):
        dispatch("mlb_competition", "baseball-mlb-competition.yml")

    if (
        not npb_ready
        and not any_active("acquisition")
        and not recently_created(data["acquisition"], now)
    ):
        dispatch("acquisition", "baseball-parallel-source-acquisition.yml")

    if (
        validation_current
        and mlb_ready
        and (not success(production) or ts(acquisition) > ts(production))
        and not any_active("production")
        and not recently_created(data["production"], now)
    ):
        dispatch("production", "baseball_production.yml")

    if (
        npb_ready
        and mlb_ready
        and validation_current
        and (not success(mac) or ts(mac) < ts(production))
        and not any_active("mac")
        and not recently_created(data["mac"], now)
    ):
        dispatch("mac", "baseball_mac_compute.yml")

    if (
        npb_ready
        and mlb_ready
        and validation_current
        and (not success(research) or ts(research) < ts(production))
        and not any_active("research")
        and not recently_created(data["research"], now)
    ):
        dispatch("research", "baseball_research.yml")

    if (
        npb_ready
        and mlb_ready
        and validation_current
        and (
            game_type_ablation is None
            or game_type_ablation.get("conclusion") in {"failure", "cancelled", "timed_out"}
            or ts(game_type_ablation) < ts(acquisition)
        )
        and not any_active("game_type_ablation")
        and not recently_created(data["game_type_ablation"], now)
    ):
        dispatch("game_type_ablation", "baseball_game_type_ablation.yml")

    diagnostic = {
        "version": "baseball-recovery-v6",
        "time": datetime.now(timezone.utc).isoformat(),
        "status": "degraded" if errors else "ok",
        "action": "cancel-stuck" if stuck else (
            "dispatch-parallel:" + ",".join(dispatched) if dispatched else "active-no-dispatch"
        ),
        "errors": errors,
        "cancelled_stuck_runs": stuck,
        "main_sha": current_sha,
        "readiness": {
            "npb": {"complete": npb_ready, "aggregate_games": npb_games},
            "mlb": {"complete": mlb_ready, "aggregate_games": mlb_games},
        },
        "active_runs": {
            label: [int(row["id"]) for row in rows]
            for label, rows in logical_active.items()
        },
        "latest": {
            label: (
                {
                    "id": int(row["id"]),
                    "status": row.get("status"),
                    "conclusion": row.get("conclusion"),
                    "created_at": row.get("created_at"),
                    "workflow_file": row.get("_workflow_file"),
                }
                if row else None
            )
            for label, row in logical_latest.items()
        },
    }
    Path("recovery_diagnostics").mkdir(parents=True, exist_ok=True)
    Path("recovery_diagnostics/latest.json").write_text(
        json.dumps(diagnostic, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    Path("recovery_diagnostics/recent_runs.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(diagnostic, ensure_ascii=False, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
