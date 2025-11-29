#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, roc_curve
from sklearn.model_selection import train_test_split

from features import add_features_pro
import joblib

DATA_DIR = "data_1m"
MODEL_DIR = "modelli_1h"    # cartella separata per i modelli 1h
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "AVAXUSDT", "XRPUSDT"]

os.makedirs(MODEL_DIR, exist_ok=True)

HORIZON_1H = 60  # 60 barre da 1m = 1 ora

# k di partenza per 1h (prima versione; poi si può fare k-sweep 1h)
K_MAP_1H = {
    "BTCUSDT": 1.0,
    "ETHUSDT": 1.0,
    "SOLUSDT": 1.2,
    "AVAXUSDT": 1.0,
    "XRPUSDT": 1.0,
}


def optimal_threshold(y_true, y_prob):
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    j = tpr - fpr
    idx = np.argmax(j)
    return thr[idx]


def train_for_symbol_1h(symbol: str):
    csv_path = os.path.join(DATA_DIR, f"{symbol}_1m.csv")
    if not os.path.exists(csv_path):
        print(f"[{symbol}][1h] CSV non trovato: {csv_path}")
        return

    print("\n=======================================")
    print(f"  TRAIN DIREZIONALE 1H: {symbol}")
    print("=======================================\n")

    # --------------------------------------------------------
    # RAW + FEATURE ENGINEERING (horizon=60, k specifico 1h)
    # --------------------------------------------------------
    df_raw = pd.read_csv(csv_path)
    k = K_MAP_1H.get(symbol, 1.0)

    df = add_features_pro(df_raw, horizon=HORIZON_1H, k=k)

    if len(df) < 5000:
        print(f"[{symbol}][1h] Troppi pochi dati dopo feature: {len(df)}")
        return

    # timestamp & filtro orario 12–21 UTC
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["hour"] = df["timestamp"].dt.hour
        df = df[df["hour"].between(12, 21)]

    # rimuovi target=0
    df = df[df["target"] != 0]
    if len(df) < 2000:
        print(f"[{symbol}][1h] Dati insufficienti dopo filtro target!=0: {len(df)}")
        return

    # target binario: +1 → 1 (LONG), -1 → 0 (SHORT)
    df["target_bin"] = (df["target"] == 1).astype(int)

    # info mapping
    mapping_path = os.path.join(MODEL_DIR, f"mapping_1h_{symbol}.txt")
    with open(mapping_path, "w") as f:
        f.write("target +1 => 1 (LONG)\n")
        f.write("target -1 => 0 (SHORT)\n")
        f.write(f"k usato (1h): {k}\n")
        f.write(f"horizon: {HORIZON_1H} minuti\n")

    # temporal dropout
    df = df.sample(frac=0.8, random_state=42).sort_values("timestamp")

    # X / y
    y = df["target_bin"]
    X = df.drop(columns=["target", "target_bin", "timestamp", "hour"], errors="ignore")

    print(f"[{symbol}][1h] Dopo cleanup: X={len(X)}, y={len(y)}")

    # split 75/25 senza shuffle
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, shuffle=False
    )

    # bilanciamento
    pos = y_train.sum()
    neg = len(y_train) - pos
    if pos == 0 or neg == 0:
        print(f"[{symbol}][1h] Una sola classe nel training (pos={pos}, neg={neg}).")
        return

    scale_pos_weight = neg / (pos + 1e-9)
    print(f"[{symbol}][1h] pos={pos}, neg={neg}, scale_pos_weight={scale_pos_weight:.2f}")

    # ============================
    # XGBOOST 1H
    # ============================
    print(f"[{symbol}][1h] Addestro XGBoost…")

    xgb = XGBClassifier(
        n_estimators=650,
        max_depth=7,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.9,
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        objective="binary:logistic",
        eval_metric="logloss",
    )
    xgb.fit(X_train, y_train)
    p_xgb = xgb.predict_proba(X_test)[:, 1]
    pred_xgb = (p_xgb > 0.5).astype(int)
    acc_xgb = accuracy_score(y_test, pred_xgb)
    print(f"[{symbol}][1h] XGB accuracy (thr=0.5): {acc_xgb:.4f}")

    # ============================
    # LIGHTGBM 1H
    # ============================
    print(f"[{symbol}][1h] Addestro LightGBM…")

    lgb = LGBMClassifier(
        n_estimators=800,
        learning_rate=0.03,
        max_depth=8,
        subsample=0.9,
        colsample_bytree=0.9,
        is_unbalance=True,
    )
    lgb.fit(X_train, y_train)
    p_lgb = lgb.predict_proba(X_test)[:, 1]
    pred_lgb = (p_lgb > 0.5).astype(int)
    acc_lgb = accuracy_score(y_test, pred_lgb)
    print(f"[{symbol}][1h] LGB accuracy (thr=0.5): {acc_lgb:.4f}")

    # ============================
    # ENSEMBLE MAX()
    # ============================
    p_final = np.maximum(p_xgb, p_lgb)
    pred_final = (p_final > 0.5).astype(int)
    acc_final = accuracy_score(y_test, pred_final)
    print(f"[{symbol}][1h] ENSEMBLE accuracy (thr=0.5): {acc_final:.4f}")

    # soglia ottimale
    thr = optimal_threshold(y_test, p_final)
    print(f"[{symbol}][1h] Soglia ottimale ensemble: {thr:.3f}")

    thr_path = os.path.join(MODEL_DIR, f"thr_1h_{symbol}.txt")
    with open(thr_path, "w") as f:
        f.write(str(thr))

    # salva modelli 1h
    joblib.dump(xgb, os.path.join(MODEL_DIR, f"xgb_1h_{symbol}.pkl"))
    joblib.dump(lgb, os.path.join(MODEL_DIR, f"lgb_1h_{symbol}.pkl"))
    print(f"[{symbol}][1h] Modelli 1h salvati in {MODEL_DIR}/")

    # qualche sample
    print("\nEsempi previsioni 1h (prime 15 righe test):")
    for i in range(min(15, len(y_test))):
        print(f" idx={i} | true={y_test.iloc[i]} | p_final={p_final[i]:.3f}")


def main():
    for sym in SYMBOLS:
        train_for_symbol_1h(sym)


if __name__ == "__main__":
    main()
