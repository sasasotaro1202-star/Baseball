#!/usr/bin/env python3
"""Research-only Asian Games baseball schedule/result collector.

Source:
  Official Site of the Japan National Baseball Team (japan-baseball.jp)

Safety:
  - No paid service.
  - Stores only schedule/result identity and source provenance.
  - Does not create prediction-time features.
  - Future/unresolved games remain score-missing and are never converted to outcomes.
  - This dataset is intentionally NOT wired into production training.

Supported official pages:
  2018 Jakarta, 2023 Hangzhou, 2026 Aichi-Nagoya.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from lxml import html as lxml_html

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results" / "international_asian_games_schedule.csv"
TIMEOUT = 30

URLS = {
    2018: "https://www.japan-baseball.jp/en/team/amateur/2018/asiangames/overview.html",
    2023: "https://www.japan-baseball.jp/en/team/amateur/2023/asiangames/overview.html",
    2026: "https://www.japan-baseball.jp/en/team/amateur/2026/asiangames/overview.html",
}


def _get(url: str) -> bytes:
    response = requests.get(
        url,
        timeout=TIMEOUT,
        headers={"User-Agent": "Baseball-International-Research/1.0"},
    )
    response.raise_for_status()
    return response.content


def _heading_before(table: Any) -> str:
    nodes = table.xpath(
        "(preceding::h1|preceding::h2|preceding::h3|preceding::h4|preceding::h5|preceding::h6)[last()]"
    )
    if not nodes:
        return ""
    return re.sub(r"\s+", " ", nodes[-1].text_content()).strip()


def _normalize_date(raw: str, edition: int) -> str | None:
    text = re.sub(r"\s+", " ", str(raw or "")).strip()
    patterns = [
        r"(?P<m>\d{1,2})/(?P<d>\d{1,2})/(?P<y>\d{4})\s+(?P<h>\d{1,2}):(?P<mi>\d{2})",
        r"(?P<y>\d{4})年(?P<m>\d{1,2})月(?P<d>\d{1,2})日\s*(?P<h>\d{1,2}):(?P<mi>\d{2})",
        r"(?P<m>\d{1,2})/(?P<d>\d{1,2})\s+(?P<h>\d{1,2}):(?P<mi>\d{2})",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        year = int(match.groupdict().get("y") or edition)
        try:
            dt = datetime(
                year,
                int(match.group("m")),
                int(match.group("d")),
                int(match.group("h")),
                int(match.group("mi")),
            )
            return dt.strftime("%Y-%m-%dT%H:%M:00+09:00")
        except ValueError:
            return None
    return None


def _parse_matchup(raw: str) -> tuple[str, str, int | None, int | None, str]:
    text = re.sub(r"\s+", " ", str(raw or "")).strip()
    if not text:
        return "", "", None, None, "UNKNOWN"
    score = re.match(r"^(.+?)\s+(\d+)\s*-\s*(\d+)\s+(.+?)$", text)
    if score:
        return (
            score.group(1).strip(),
            score.group(4).strip(),
            int(score.group(2)),
            int(score.group(3)),
            "FINAL",
        )
    future = re.match(r"^(.+?)\s+-\s+(.+?)$", text)
    if future:
        return future.group(1).strip(), future.group(2).strip(), None, None, "SCHEDULED"
    return "", "", None, None, "UNKNOWN"


def parse_overview_html(content: bytes, edition: int, source_url: str, retrieved_at: str) -> pd.DataFrame:
    """Parse tournament-wide schedule tables from an official overview page."""
    doc = lxml_html.fromstring(content)
    rows: list[dict[str, Any]] = []
    tables = doc.xpath('//table[.//th[contains(normalize-space(.), "Home - Visitor") or contains(normalize-space(.), "対戦")]]')
    for table_index, table in enumerate(tables):
        phase = _heading_before(table) or "UNKNOWN"
        tr_nodes = table.xpath(".//tr")
        for row_index, tr in enumerate(tr_nodes):
            cells = [
                re.sub(r"\s+", " ", cell.text_content()).strip()
                for cell in tr.xpath("./th|./td")
            ]
            if len(cells) < 3:
                continue
            date_text, venue, matchup = cells[0], cells[1], cells[2]
            dt = _normalize_date(date_text, edition)
            home, away, hs, aas, status = _parse_matchup(matchup)
            if not home or not away or not dt:
                continue
            identity = f"asian_games|{edition}|{dt}|{home}|{away}|{venue}"
            game_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
            rows.append(
                {
                    "league": "INTERNATIONAL",
                    "competition": "Asian Games",
                    "international_game_category": "asian_games",
                    "edition": int(edition),
                    "phase": phase,
                    "game_id": game_id,
                    "datetime_jst": dt,
                    "home": home,
                    "away": away,
                    "home_score": hs,
                    "away_score": aas,
                    "status": status,
                    "venue": venue,
                    "source_url": source_url,
                    "source_retrieved_at": retrieved_at,
                    "prediction_features_allowed": False,
                    "pit_prediction_status": "NOT_A_PREDICTION_FEATURE_SOURCE",
                    "table_index": int(table_index),
                    "row_index": int(row_index),
                }
            )
    if not rows:
        raise RuntimeError(f"No Asian Games schedule rows parsed from {source_url}")
    out = pd.DataFrame(rows)
    out = out.drop_duplicates("game_id").sort_values(["datetime_jst", "game_id"]).reset_index(drop=True)
    return out


def collect(editions: list[int]) -> pd.DataFrame:
    retrieved_at = datetime.now(timezone.utc).isoformat()
    frames = []
    for edition in editions:
        url = URLS[edition]
        content = _get(url)
        frames.append(parse_overview_html(content, edition, url, retrieved_at))
    return pd.concat(frames, ignore_index=True).drop_duplicates("game_id").sort_values(
        ["datetime_jst", "game_id"]
    ).reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--editions", default="2018,2023,2026")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    editions = []
    for raw in str(args.editions).split(","):
        value = int(raw.strip())
        if value not in URLS:
            raise SystemExit(f"unsupported edition: {value}")
        editions.append(value)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df = collect(editions)
    df.to_csv(output, index=False)

    manifest = {
        "schema_version": 1,
        "status": "PASS",
        "source": "japan-baseball.jp",
        "editions": editions,
        "rows": int(len(df)),
        "final_rows": int((df["status"] == "FINAL").sum()),
        "scheduled_rows": int((df["status"] == "SCHEDULED").sum()),
        "prediction_features_allowed": False,
        "production_training_enabled": False,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "output": str(output),
    }
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
