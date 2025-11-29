#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from features_15m import add_features_pro

DATA_DIR = "data_1m"
MODEL_DIR = "modelli_tuned"
os.makedirs(MODEL_DIR, exist_ok=True)

SYMBOL = "BTCUSDT"   # puoi cambiarlo o iterare


# ============================================================
# Soglia ottimale (ROC Youden)
# ============================================================
def optimal_threshold(y_true, p):
    fpr, tpr, thr = roc_curve(y_true, p)
    j = tpr - fpr
    return thr[np.argmax(j)]


# ============================================================
# Tuning param grid (ridotto ma efficace)
# ============================================================
param_grid_xgb = [
    {
        "n_estimators": n,
        "max_depth": d,
        "learning_rate": lr,
        "subsample": ss,
        "colsample_bytree": cs
    }
    for n in [300, 500, 700]
    for d in [5, 6, 7]
    for lr in [0.03, 0.05]
    for ss in [0.8, 0.9]
    for cs in [0.8, 0.9]
]

param_grid_lgb = [
    {
        "n_estimators": n,
        "max_depth": d,
        "learning_rate": lr,
        "subsample": ss,
        "colsample_bytree": cs
    }
    for n in [300, 500, 700]
    for d in [-1, 7]
    for lr in [0.03, 0.05]
    for ss in [0.8, 0.9]
    for cs in [0.8, 0.9]
]


# ============================================================
# Carica dati + feature + filtro orario
# ============================================================
def load_dataset(symbol):
    path = os.path.join(DATA_DIR, f"{symbol}_1m.csv")
    df_raw = pd.read_csv(path)

    df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"])
    df_raw = df_raw.sort_values("timestamp")

    df = add_features_pro(df_raw)

    # Filtro orario 12–21 (regimi profittevoli)
    df["hour"] = df["timestamp"].dt.hour
    df = df[df["hour"].between(12, 21)]

    # Target separato
    y = df["target"]
    X = df.drop(columns=["target", "timestamp"])

    # Train/test split temporale
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, shuffle=False
    )

    return X_train, X_test, y_train, y_test


# ============================================================
# Tuning XGB
# ============================================================
def tune_xgb(X_train, X_test, y_train, y_test):
    best_score = -1
    best_params = None
    best_model = None

    pos = y_train.sum()
    neg = len(y_train) - pos
    spw = neg / (pos + 1e-9)

    print("\n=====================")
    print("TUNING XGBOOST")
    print("=====================\n")

    for params in param_grid_xgb:
        model = XGBClassifier(
            **params,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=spw,
            tree_method="hist",
        )

        model.fit(X_train, y_train)
        p = model.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, p)

        if auc > best_score:
            best_score = auc
            best_params = params
            best_model = model

        print(f"AUC={auc:.4f}   params={params}")

    print("\nBest XGB AUC:", best_score)
    print("Best params:", best_params)
    return best_model, best_params


# ============================================================
# Tuning LGB
# ============================================================
def tune_lgb(X_train, X_test, y_train, y_test):
    best_score = -1
    best_params = None
    best_model = None

    print("\n=====================")
    print("TUNING LIGHTGBM")
    print("=====================\n")

    for params in param_grid_lgb:
        model = LGBMClassifier(
            **params,
            is_unbalance=True,
        )

        model.fit(X_train, y_train)
        p = model.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, p)

        if auc > best_score:
            best_score = auc
            best_params = params
            best_model = model

        print(f"AUC={auc:.4f}   params={params}")

    print("\nBest LGB AUC:", best_score)
    print("Best params:", best_params)
    return best_model, best_params


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":

    print("\n==========================")
    print(" TUNING ENSEMBLE 15M")
    print("==========================")

    X_train, X_test, y_train, y_test = load_dataset(SYMBOL)

    best_xgb, best_xgb_params = tune_xgb(X_train, X_test, y_train, y_test)
    best_lgb, best_lgb_params = tune_lgb(X_train, X_test, y_train, y_test)

    p_final = np.maximum(
        best_xgb.predict_proba(X_test)[:, 1],
        best_lgb.predict_proba(X_test)[:, 1]
    )

    ensemble_auc = roc_auc_score(y_test, p_final)
    thr = optimal_threshold(y_test, p_final)

    print("\n==========================")
    print(f" AUC FINALE ENSEMBLE: {ensemble_auc:.4f}")
    print(f" SOGLIA OTTIMALE: {thr:.4f}")
    print("==========================\n")

    # Salva modelli e soglia
    joblib.dump(best_xgb, os.path.join(MODEL_DIR, f"xgb_15m_{SYMBOL}.pkl"))
    joblib.dump(best_lgb, os.path.join(MODEL_DIR, f"lgb_15m_{SYMBOL}.pkl"))

    with open(os.path.join(MODEL_DIR, f"thr_15m_{SYMBOL}.txt"), "w") as f:
        f.write(str(thr))

    print("\nModelli salvati correttamente.")
