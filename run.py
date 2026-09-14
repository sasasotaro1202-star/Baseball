#!/usr/bin/env python3
"""Main entry point for sports prediction pipeline."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def cmd_collect_baseball(args):
    """Collect MLB data from StatsAPI."""
    sys.path.insert(0, str(ROOT))
    from pipelines.collect import collect_mlb
    collect_mlb()
    return 0


def cmd_features(args):
    """Build Elo baseline + base rate comparison."""
    result = subprocess.run([sys.executable, str(ROOT / "elo_baseline_all.py")], cwd=str(ROOT))
    return result.returncode


def cmd_research(args):
    """Alias for features."""
    return cmd_features(args)


def cmd_predict(args):
    """Placeholder until the production predictor is invoked by its dedicated workflow."""
    print("predict: no standalone staged model command; production workflow owns prediction")
    return 0


def cmd_verify(args):
    """Placeholder until the production verifier is invoked by its dedicated workflow."""
    print("verify: no standalone verifier; production workflow owns verification")
    return 0


def cmd_inventory(args):
    """Show data inventory by scanning data/ directly."""
    data_dir = ROOT / "data"
    if not data_dir.exists():
        print("No data/ directory yet.")
        return 0

    for layer_dir in sorted(data_dir.iterdir()):
        if not layer_dir.is_dir():
            continue
        files = list(layer_dir.rglob("*.csv")) + list(layer_dir.rglob("*.parquet")) + list(layer_dir.rglob("*.json"))
        if files:
            print(f"{layer_dir.name}: {len(files)} files")
            for f in files[:10]:
                print(f"  - {f.relative_to(data_dir)} ({f.stat().st_size} bytes)")
            if len(files) > 10:
                print(f"  ... and {len(files) - 10} more")
    return 0


def cmd_shard_plan(args):
    """Emit a valid, deterministic matrix for the parallel MLB planning workflow.

    This planner is intentionally limited to baseball. It does not claim that
    collection has happened; matrix workers remain responsible for their work.
    """
    import os

    # One shard per configured MLB season keeps the plan deterministic and makes
    # retries idempotent. The collector/backtest workflows remain authoritative.
    start = int(os.environ.get("MLB_PARALLEL_START_YEAR", "2020"))
    end = int(os.environ.get("MLB_PARALLEL_END_YEAR", "2026"))
    if start > end or end - start > 30:
        raise ValueError(f"invalid MLB shard range: {start}-{end}")

    include = [
        {"source": "mlb", "index": str(i), "total": str(end - start + 1), "season": str(year)}
        for i, year in enumerate(range(start, end + 1), start=1)
    ]
    matrix = {"include": include}
    payload = json.dumps(matrix, separators=(",", ":"))
    if args.emit_matrix:
        output_file = os.environ.get("GITHUB_OUTPUT")
        if not output_file:
            raise RuntimeError("--emit-matrix requires GITHUB_OUTPUT")
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(f"matrix={payload}\n")
    else:
        print(payload)
    return 0


def build_parser():
    p = argparse.ArgumentParser(prog="run.py")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("collect-baseball")
    sp.set_defaults(func=cmd_collect_baseball)

    sp = sub.add_parser("features")
    sp.set_defaults(func=cmd_features)

    sp = sub.add_parser("research")
    sp.set_defaults(func=cmd_research)

    sp = sub.add_parser("predict")
    sp.set_defaults(func=cmd_predict)

    sp = sub.add_parser("verify")
    sp.set_defaults(func=cmd_verify)

    sp = sub.add_parser("inventory")
    sp.set_defaults(func=cmd_inventory)

    sp = sub.add_parser("shard-plan")
    sp.add_argument("--emit-matrix", action="store_true")
    sp.set_defaults(func=cmd_shard_plan)

    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args) or 0
    except Exception as e:
        print(f"Command '{args.command}' failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
