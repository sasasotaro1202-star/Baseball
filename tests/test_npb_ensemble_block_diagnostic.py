import numpy as np
import pandas as pd

from baseball_backtest import BaseballBacktest


def test_diagnose_npb_ensemble_block_2339() -> None:
    bt = BaseballBacktest()
    games = bt.aggregate_npb_games(bt.load_npb_pbp())
    X, y, meta = bt.build_features(games)
    train_mask = meta["training_included"].astype(bool).to_numpy()
    bstart, bend = 2339, 2489
    assert len(X) >= bend

    X_train = X.iloc[:bstart][train_mask[:bstart]]
    y_train = y[:bstart][train_mask[:bstart]]
    assert len(X_train) >= 100
    assert len(np.unique(y_train)) >= 3

    models = bt.models("NPB")
    failures = {}
    successes = []
    splits = bt._validation_splits(len(X_train))
    for name, model in models.items():
        model_losses = []
        try:
            for cut, val in splits:
                Xfit, Xval = X_train.iloc[:cut], X_train.iloc[cut:cut + val]
                yfit, yval = y_train[:cut], y_train[cut:cut + val]
                bt._fit_model(model, Xfit, yfit, bt._sample_weights(len(Xfit)), "NPB")
                p = bt.align_proba(model.predict_proba(Xval), model.classes_, "NPB")
                from sklearn.metrics import log_loss
                model_losses.append(float(log_loss(yval, p, labels=[0, 1, 2])))
            if model_losses:
                successes.append((name, float(np.mean(model_losses))))
            else:
                failures[name] = "no validation losses"
        except Exception as exc:
            failures[name] = f"{type(exc).__name__}: {exc}"
    print("NPB_BLOCK_2339_MODEL_SUCCESSES", successes)
    print("NPB_BLOCK_2339_MODEL_FAILURES", failures)
    assert successes, f"all candidate models failed: {failures}"
