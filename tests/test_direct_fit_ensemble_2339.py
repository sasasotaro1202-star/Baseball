import numpy as np

from baseball_backtest import BaseballBacktest


def test_direct_fit_ensemble_2339_diagnostic():
    bt = BaseballBacktest()
    games = bt.aggregate_npb_games(bt.load_npb_pbp())
    X, y, meta = bt.build_features(games)
    mask = meta['training_included'].astype(bool).to_numpy()
    bstart = 2339
    X_train = X.iloc[:bstart][mask[:bstart]]
    y_train = y[:bstart][mask[:bstart]]
    result = bt.fit_ensemble(X_train, y_train, 'NPB')
    fitted, scores, best = result
    raise AssertionError(
        f'DIRECT_FIT_ENSEMBLE_2339 fitted={bool(fitted)} scores={scores} best={best} n={len(X_train)} classes={np.unique(y_train).tolist()}'
    )