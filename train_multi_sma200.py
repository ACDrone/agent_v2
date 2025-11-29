#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_sma200_1d.py

Modello "SMA200 DAILY" (regime lungo):
- candele 1d (data_1d/<symbol>_1d.csv)
- target: tra H_SMA_DAYS giorni saremo ben SOPRA o SOTTO la SMA200 di ADESSO,
  confrontato con ATR daily * k.

Serve come filtro di regime più "macro" rispetto all'EMA50.
"""

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import joblib

from features_long_1d import add_features_daily

# ==============================
# CONFIG
# ==============================
DATA_DIR = "data_1d"
MODEL_DIR = "modelli_sma200"
os.makedirs(MODEL_DIR, exist_ok=True)

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "AVAXUSDT", "XRPUSDT"]

# orizzonte in giorni (più lungo dell'EMA50)
H_SMA_DAYS = 30
HORIZON_SMA_BARS = H_SMA_DAYS

K_MAP_SMA = {
    "BTCUSDT": 1.2,
    "ETHUSDT": 1.2,
    "SOLUSDT": 1.5,
    "AVAXUSDT": 1.3,
    "XRPUSDT": 1.3,
}


def optimal_threshold(y_true, y_prob):
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    j = tpr - fpr
    idx = np.argmax(j)
    return thr[idx]


def add_target_sma200_1d(df_raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Aggiunge colonna target_sma200_1d:

      +1 se tra H_SMA_DAYS giorni close_future > sma200_now + ATR14_daily*k
      -1 se tra H_SMA_DAYS giorni close_future < sma200_now - ATR14_daily*k
       0 altrimenti
    """
    df = df_raw.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # SMA200 daily
    df["sma200_d"] = df["close"].rolling(200).mean()

    # ATR 14 daily
    df["H-L_d"] = df["high"] - df["low"]
    df["H-PC_d"] = (df["high"] - df["close"].shift(1)).abs()
    df["L-PC_d"] = (df["low"] - df["close"].shift(1)).abs()
    df["TR_d"] = df[["H-L_d", "H-PC_d", "L-PC_d"]].max(axis=1)
    df["atr14_d"] = df["TR_d"].rolling(14).mean()

    # close futuro a H_SMA_DAYS
    df["future_close_sma"] = df["close"].shift(-HORIZON_SMA_BARS)

    k = K_MAP_SMA.get(symbol, 1.2)
    df["thr_sma"] = df["atr14_d"] * k

    df["target_sma200_1d"] = 0
    df.loc[df["future_close_sma"] > df["sma200_d"] + df["thr_sma"], "target_sma200_1d"] = 1
    df.loc[df["future_close_sma"] < df["sma200_d"] - df["thr_sma"], "target_sma200_1d"] = -1

    keep_cols = ["timestamp", "open", "high", "low", "close", "volume", "target_sma200_1d"]
    return df[keep_cols]


def train_for_symbol_sma200_1d(symbol: str):
    csv_path = os.path.join(DATA_DIR, f"{symbol}_1d.csv")
    if not os.path.exists(csv_path):
        print(f"[{symbol}][SMA200_1D] CSV non trovato: {csv_path}")
        return

    print("\n====================================================")
    print(f"  TRAIN SMA200 DAILY (H={H_SMA_DAYS}d): {symbol}")
    print("====================================================\n")

    df_raw = pd.read_csv(csv_path)
    df_lbl = add_target_sma200_1d(df_raw, symbol)

    df_feat = add_features_daily(df_lbl)

    if "target_sma200_1d" not in df_feat.columns:
        print(f"[{symbol}][SMA200_1D] target_sma200_1d non presente dopo le feature.")
        return

    df = df_feat[df_feat["target_sma200_1d"] != 0].copy()
    if len(df) < 1000:
        print(f"[{symbol}][SMA200_1D] pochi dati dopo filtro target!=0: {len(df)}")
        return

    df["target_bin"] = (df["target_sma200_1d"] == 1).astype(int)

    df = df.sort_values("timestamp")
    y = df["target_bin"]
    X = df.drop(columns=["target_sma200_1d", "target_bin"], errors="ignore")
    if "timestamp" in X.columns:
        X = X.drop(columns=["timestamp"])

    print(f"[{symbol}][SMA200_1D] Dopo cleanup: X={len(X)}, y={len(y)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, shuffle=False
    )

    pos = y_train.sum()
    neg = len(y_train) - pos
    if pos == 0 or neg == 0:
        print(f"[{symbol}][SMA200_1D] Una sola classe nel train (pos={pos}, neg={neg}).")
        return

    scale_pos_weight = neg / (pos + 1e-9)
    print(f"[{symbol}][SMA200_1D] pos={pos}, neg={neg}, scale_pos_weight={scale_pos_weight:.2f}")

    print(f"[{symbol}][SMA200_1D] Addestro XGBoost…")
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
        n_jobs=-1,
    )
    xgb.fit(X_train, y_train)
    p_xgb = xgb.predict_proba(X_test)[:, 1]
    acc_xgb = accuracy_score(y_test, (p_xgb > 0.5).astype(int))
    print(f"[{symbol}][SMA200_1D] XGB accuracy (thr=0.5): {acc_xgb:.4f}")

    print(f"[{symbol}][SMA200_1D] Addestro LightGBM…")
    lgb = LGBMClassifier(
        n_estimators=800,
        learning_rate=0.03,
        max_depth=8,
        subsample=0.9,
        colsample_bytree=0.9,
        is_unbalance=True,
        n_jobs=-1,
    )
    lgb.fit(X_train, y_train)
    p_lgb = lgb.predict_proba(X_test)[:, 1]
    acc_lgb = accuracy_score(y_test, (p_lgb > 0.5).astype(int))
    print(f"[{symbol}][SMA200_1D] LGB accuracy (thr=0.5): {acc_lgb:.4f}")

    p_final = np.maximum(p_xgb, p_lgb)
    acc_final = accuracy_score(y_test, (p_final > 0.5).astype(int))
    print(f"[{symbol}][SMA200_1D] ENSEMBLE accuracy (thr=0.5): {acc_final:.4f}")

    thr = optimal_threshold(y_test, p_final)
    print(f"[{symbol}][SMA200_1D] Soglia ottimale ensemble: {thr:.3f}")

    joblib.dump(xgb, os.path.join(MODEL_DIR, f"xgb_sma200_1d_{symbol}.pkl"))
    joblib.dump(lgb, os.path.join(MODEL_DIR, f"lgb_sma200_1d_{symbol}.pkl"))

    with open(os.path.join(MODEL_DIR, f"thr_sma200_1d_{symbol}.txt"), "w") as f:
        f.write(str(thr))

    print(f"[{symbol}][SMA200_1D] Modelli SMA200 DAILY salvati in {MODEL_DIR}/")

    print("\nEsempi previsioni SMA200 DAILY (prime 15 righe test):")
    for i in range(min(15, len(y_test))):
        print(f" idx={i} | true={y_test.iloc[i]} | p_final={p_final[i]:.3f}")


def main():
    for sym in SYMBOLS:
        train_for_symbol_sma200_1d(sym)


if __name__ == "__main__":
    main()
