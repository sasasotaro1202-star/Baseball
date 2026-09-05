# Mac Compute Worker

This directory documents the one-time registration of the user's Mac as a GitHub Actions self-hosted runner.

## Target architecture

GitHub is the source of truth and orchestration layer. The Mac is the high-throughput compute worker for expensive chronological/OOS baseball backtests and research.

```text
Baseball Production Backtest (GitHub)
        |
        v
persisted data/results on main
        |
        v
Baseball Mac Compute Research
        |
        |  self-hosted macOS/ARM64 runner
        v
Mac: deep NPB + MLB OOS computation
        |
        v
research state/results committed to main
        |
        v
Baseball Autonomous Research
        |
        v
weakness analysis -> research priority -> next experiment
```

## One-time local requirement

GitHub cannot remotely install/register a self-hosted runner on a Mac from repository contents alone. The Mac must be registered once from GitHub's **Settings -> Actions -> Runners** page using the runner commands GitHub supplies for that repository.

Use the labels:

- `self-hosted`
- `macOS`
- `ARM64`

After registration, leave the runner service installed so it starts automatically with macOS. No routine manual execution is required.

Do not commit a runner token, PAT, SSH private key, or other credential to this repository.

## Expected Python

The worker creates an isolated `.venv` on each run and installs `requirements.txt`. Python 3.11 is recommended for parity with the cloud workflows.

## Safety contract

The Mac worker must preserve the same source-quality, chronological/OOS, starter-coverage, score/Low-High, and leakage controls as production. A failed or incomplete experiment must never replace the current model merely because it ran on faster hardware.
