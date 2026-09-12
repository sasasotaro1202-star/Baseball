import pytest

from data.market_lines import TotalRunsLine
from prediction.baseball_prediction_adapter import build_low_high, build_runner_row, build_score_candidates


def test_score_candidates_are_exact_top_four_and_not_fake_tail():
    rows = build_score_candidates(4.0, 3.0)
    assert len(rows) == 4
    assert all("score" in r and "probability" in r for r in rows)
    assert all("その他" not in r["score"] for r in rows)
    assert all(0.0 <= r["probability"] <= 1.0 for r in rows)
    assert rows == sorted(rows, key=lambda r: (-r["probability"], r["score"]))


def test_low_high_requires_verified_half_point_line():
    line = TotalRunsLine(
        event_id="g1", league="MLB", line=7.5, source="test",
        observed_at="2026-09-12T00:00:00+00:00", available_at="2026-09-12T00:00:00+00:00",
    )
    low, high = build_low_high(4.0, 3.0, line)
    assert low is not None and high is not None
    assert abs(low + high - 1.0) < 1e-8


def test_integer_line_does_not_fake_two_way_low_high():
    line = TotalRunsLine(
        event_id="g2", league="NPB", line=7.0, source="test",
        observed_at="2026-09-12T00:00:00+00:00", available_at="2026-09-12T00:00:00+00:00",
    )
    assert build_low_high(4.0, 3.0, line) == (None, None)


def test_runner_row_rejects_line_not_available_by_cutoff():
    line = TotalRunsLine(
        event_id="g3", league="MLB", line=7.5, source="test",
        observed_at="2026-09-12T02:00:00+00:00", available_at="2026-09-12T02:00:00+00:00",
    )
    with pytest.raises(ValueError):
        build_runner_row(
            probabilities={"home": 0.6, "away": 0.4},
            lam_home=4.0, lam_away=3.0,
            total_line=line,
            cutoff="2026-09-12T01:00:00+00:00",
        )
