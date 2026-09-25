#!/usr/bin/env python3
from pathlib import Path
import os
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from baseball_backtest import BaseballBacktest, MIN_TRAIN, MIN_VALIDATION

DATA = Path("data")
df = pd.read_csv(DATA / "mlb_games.csv", low_memory=False)
df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
df = df.sort_values(["datetime","game_id"]).reset_index(drop=True)
print("[DEBUG] rows", len(df))
print("[DEBUG] categories", df["mlb_game_category"].value_counts(dropna=False).to_dict())

bt = BaseballBacktest(DATA)
X, y, meta = bt.build_features(df)
train_mask = meta["training_included"].astype(bool).to_numpy()
eval_mask = meta["evaluation_included"].astype(bool).to_numpy()
print("[DEBUG] X shape", X.shape)
print("[DEBUG] X numeric", int(sum(pd.api.types.is_numeric_dtype(X[c]) for c in X.columns)), "/", X.shape[1])
print("[DEBUG] nonfinite cells", int((~np.isfinite(X.to_numpy(dtype=float))).sum()))
for col in X.columns:
    arr=pd.to_numeric(X[col], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(arr).all():
        print("[DEBUG] nonfinite column", col, "count", int((~np.isfinite(arr)).sum()))

bstart=303
cut_windows=bt._validation_splits(bstart)
print("[DEBUG] validation_splits", cut_windows)
X_train = X.iloc[:bstart][train_mask[:bstart]]
y_train = y[:bstart][train_mask[:bstart]]
print("[DEBUG] effective train", len(y_train), "classes", np.unique(y_train, return_counts=True))

models=bt.models("MLB")
k=2
for name, model in models.items():
    losses=[]
    print("\n[MODEL]", name)
    for cut,val in bt._validation_splits(len(X_train)):
        Xfit,Xval=X_train.iloc[:cut],X_train.iloc[cut:cut+val]
        yfit,yval=y_train[:cut],y_train[cut:cut+val]
        print(" split",cut,val,"classes",np.unique(yfit,return_counts=True))
        try:
            bt._fit_model(model,Xfit,yfit,bt._sample_weights(cut),"MLB")
            p=bt.align_proba(model.predict_proba(Xval),model.classes_,"MLB")
            ll=log_loss(yval,p,labels=[0,1])
            losses.append(float(ll))
            print("  PASS logloss",ll,"pfinite",bool(np.isfinite(p).all()))
        except Exception as e:
            print("  FAIL",type(e).__name__,repr(str(e)))
            losses=[]
            break
    print(" result losses",losses)
