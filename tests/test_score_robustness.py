import pandas as pd
from baseball_backtest import BaseballBacktest, WIN_PRIOR


def test_win_rate_shrinkage_small_sample_stays_near_neutral():
    bt = BaseballBacktest()
    state = bt.state("NPB", "テストチーム")
    state.results.extend([0, 2, 2, 0])
    state.total_matches = 4
    state.home_results.extend([0, 2])
    state.home_matches = 2
    f = bt._team_features("NPB", "テストチーム", "home", pd.Timestamp("2026-09-25", tz="UTC"))
    assert 0.0 <= f["win_rate_shrunk_all"] <= 1.0
    assert abs(f["win_rate_shrunk_all"] - WIN_PRIOR) < 0.02
    assert 0.0 <= f["venue_win_rate_shrunk"] <= 1.0


def test_negative_binomial_challenger_runs_without_oos_artifacts():
    from research import negative_binomial_score_oos as nb
    assert 0.0 < nb.FIT_FRAC < 1.0
    assert nb.MIN_ROWS >= 100

from baseball_backtest import low_high_probs, score_candidates


def test_score_candidates_are_top_four_concrete_scores():
    choices = score_candidates(2.7, 2.1, 4)
    assert len(choices) == 4
    assert all(score != "その他" for score, _ in choices)
    assert all("-" in score for score, _ in choices)
    probs = [prob for _, prob in choices]
    assert probs == sorted(probs, reverse=True)


def test_score_candidates_can_naturally_include_seven_plus():
    choices = score_candidates(8.0, 8.0, 4)
    assert len(choices) == 4
    assert any(
        int(score.split("-", 1)[0]) >= 7 or int(score.split("-", 1)[1]) >= 7
        for score, _ in choices
    )


def test_low_high_maps_seven_plus_to_high():
    low, high = low_high_probs(7.0, 1.0)
    assert low < 0.5
    assert high > 0.5
