#!/usr/bin/env python3
"""Main entry point for sports prediction pipeline"""
import argparse
import sys
from pathlib import Path

def cmd_collect_baseball(args):
    """Collect MLB data from StatsAPI"""
    from pipelines.collect import collect_mlb
    collect_mlb()
    return 0

def cmd_features(args):
    """Build features from collected data"""
    from elo_baseline_all import main as elo_main
    elo_main()
    return 0

def cmd_research(args):
    """Run backtest research"""
    print("Research not yet implemented - use elo_baseline_all.py directly")
    return 0

def cmd_predict(args):
    """Run predictions"""
    print("Predict not yet implemented")
    return 0

def cmd_verify(args):
    """Verify predictions"""
    print("Verify not yet implemented")
    return 0

def cmd_inventory(args):
    """Show data inventory"""
    from core import storage
    for layer in ["raw", "events", "features", "results", "predictions"]:
        datasets = storage.datasets(layer)
        if datasets:
            print(f"{layer}: {len(datasets)} datasets")
            for ds in datasets[:5]:
                count = storage.row_count(layer, ds)
                print(f"  - {ds}: {count} rows")
            if len(datasets) > 5:
                print(f"  ... and {len(datasets)-5} more")
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
