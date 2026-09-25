import pandas as pd
import pytest

from mlb_incremental_acquire import incomplete_historical_years, validate_historical_coverage


def _rows(counts):
    rows=[]
    for year,count in counts.items():
        for i in range(count):
            rows.append({
                "game_id": f"{year}-{i}",
                "datetime": f"{year}-07-01T12:00:00Z",
                "home_score": 1,
                "away_score": 0,
                "home": "Home",
                "away": "Away",
            })
    return pd.DataFrame(rows)


def test_incomplete_historical_year_is_detected():
    d=_rows({2020:100,2021:2100})
    assert incomplete_historical_years(d,2020,2021) == [2020]


def test_historical_coverage_passes_for_known_minimums():
    d=_rows({2020:800,2021:2000,2022:2000,2023:2000,2024:2000,2025:2000})
    validate_historical_coverage(d,2020,2025)


def test_historical_coverage_fails_closed_when_missing():
    d=_rows({2020:799,2021:2000,2022:2000,2023:2000,2024:2000,2025:2000})
    with pytest.raises(RuntimeError, match="historical coverage incomplete"):
        validate_historical_coverage(d,2020,2025)
