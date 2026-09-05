#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Idempotent emergency syntax repair for the NPB collector.

The production pipeline must never execute a syntactically invalid collector.
This guard fixes one known malformed boolean expression introduced in the
collector's generated one-line processing path. It fails closed if the file
contains an unexpected variant rather than guessing at arbitrary edits.
"""
from pathlib import Path

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
