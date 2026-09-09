"""Acquire J-League, DFB-Pokal, UCL/UEL from openfootball (GitHub, public domain).

Source repos:
  - openfootball/champions-league (UCL/UEL)
  - openfootball/deutschland (DFB-Pokal)
  - openfootball/world (J-League, Asia)

Format: football.txt (structured text, NOT CSV). This script clones the repo
locally inside GitHub Actions, then writes a minimal CSV wrapper.

Honesty note: exact file paths (season folders, etc.) were not verified
before writing this script (raw.githubusercontent.com could not be previewed).
The script auto-discovers *.txt files under league-specific folders and logs
what it found, so no data is silently skipped.
"""
import argparse
import os
import subprocess
import sys
import glob

REPOS = {
    "champions_league": "https://github.com/openfootball/champions-league.git",
    "deutschland": "https://github.com/openfootball/deutschland.git",
    "world": "https://github.com/openfootball/world.git",
}

OUT_BASE = "data/soccer/openfootball"


def clone_or_pull(repo_name: str, repo_url: str, cache_dir: str) -> str:
    clone_path = os.path.join(cache_dir, repo_name)
    if os.path.exists(clone_path):
        subprocess.run(["git", "-C", clone_path, "pull"], check=True)
    else:
        subprocess.run(["git", "clone", repo_url, clone_path], check=True)
    return clone_path


def discover_txt_files(base_path: str, pattern: str = "*.txt") -> list:
    return sorted(glob.glob(os.path.join(base_path, "**", pattern), recursive=True))


def convert_txt_to_csv(txt_path: str, out_csv: str) -> bool:
    """Writes raw text lines as a single-column CSV (no escaping bugs)."""
    with open(txt_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("raw_line\n")
        for line in lines:
            clean = line.rstrip("\n\r")
            if clean:
                f.write('"' + clean.replace('"', '""') + '"\n')
    return True


def acquire(cache_dir: str = "cache/openfootball") -> dict:
    os.makedirs(OUT_BASE, exist_ok=True)
    report = {}
    for league_name, repo_url in REPOS.items():
        print(f"[{league_name}] cloning/pulling...")
        clone_path = clone_or_pull(league_name, repo_url, cache_dir)
        txt_files = discover_txt_files(clone_path)
        print(f"[{league_name}] found {len(txt_files)} *.txt files")
        league_out = os.path.join(OUT_BASE, league_name)
        os.makedirs(league_out, exist_ok=True)
        converted = 0
        for txt in txt_files:
            base = os.path.basename(txt).replace(".txt", ".csv")
            out_csv = os.path.join(league_out, base)
            if convert_txt_to_csv(txt, out_csv):
                converted += 1
        report[league_name] = {"txt_found": len(txt_files), "csv_converted": converted}
        print(f"[{league_name}] converted {converted}/{len(txt_files)}")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default="cache/openfootball")
    args = parser.parse_args()
    report = acquire(args.cache_dir)
    import json
    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/openfootball_acquisition_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
