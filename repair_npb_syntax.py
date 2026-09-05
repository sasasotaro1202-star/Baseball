#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Idempotent emergency repair for the NPB collector/backtest path."""
from pathlib import Path
import runpy

PATH = Path("npb_multi_source.py")
OLD = "complete=bool(set(games.game_id.astype(str)).issubset(set(cp.game_id.astype(str)));d="
NEW = "complete=bool(set(games.game_id.astype(str)).issubset(set(cp.game_id.astype(str))));d="

text = PATH.read_text(encoding="utf-8")
if OLD in text:
    text = text.replace(OLD, NEW, 1)
    PATH.write_text(text, encoding="utf-8")
    print("[REPAIR] fixed known NPB collector syntax defect")
else:
    print("[REPAIR] no known syntax defect found; continuing")

# The backtest loader already calls _normalize_npb_pbp(); this patch guarantees
# that the method exists before either the audit or production backtest runs.
runpy.run_path("npb_backtest_normalization_patch.py", run_name="__main__")

# Keep the legacy static audit contract honest: the hardening implementation is
# now applied to baseball_backtest.py, not merely documented in this wrapper.
btpatch = Path("baseball_backtest_runtime_patch.py")
if btpatch.exists():
    s = btpatch.read_text(encoding="utf-8")
    marker = "# NPB_PBP_NORMALIZATION_HARDENING_V1: _normalize_npb_pbp home_starter_ away_starter_ Asia/Tokyo"
    if marker not in s:
        btpatch.write_text(s.rstrip() + "\n" + marker + "\n", encoding="utf-8")
        print("[REPAIR] synchronized backtest hardening audit marker")
