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


def test_score_candidates_are_exactly_four_choices_with_other_bucket():
    choices = score_candidates(2.7, 2.1, 4)
    assert len(choices) == 4
    assert sum(1 for score, _ in choices if score == "その他") == 1
    assert sum(1 for score, _ in choices if score != "その他") == 3


def test_low_high_maps_seven_plus_to_high():
    low, high = low_high_probs(7.0, 1.0)
    assert low < 0.5
    assert high > 0.5
