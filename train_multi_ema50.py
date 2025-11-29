#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_ema50_1d.py

Modello "EMA50 DAILY" (regime swing):
- lavora su candele 1d (data_1d/<symbol>_1d.csv)
- target: tra H_EMA_DAYS giorni saremo ben SOPRA o SOTTO l'EMA50 di ADESSO,
  confrontato con ATR daily * k.

Serve come modello di bias/regime su orizzonte ~10 giorni.
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
MODEL_DIR = "modelli_ema50"
os.makedirs(MODEL_DIR, exist_ok=True)

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "AVAXUSDT", "XRPUSDT"]

# orizzonte in GIORNI
H_EMA_DAYS = 10       # fra 10 giorni
HORIZON_EMA_BARS = H_EMA_DAYS  # 1 barra = 1d

# intensità soglia ATR * k
K_MAP_EMA = {
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


def add_target_ema50_1d(df_raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Aggiunge colonna target_ema50_1d:

      +1 se tra H_EMA_DAYS giorni close_future > ema50_now + ATR14_daily*k
      -1 se tra H_EMA_DAYS giorni close_future < ema50_now - ATR14_daily*k
       0 altrimenti
    """

    df = df_raw.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # EMA50 daily
    df["ema50_d"] = df["close"].ewm(span=50, adjust=False).mean()

    # ATR 14 daily (lo rifacciamo qui per la soglia)
    df["H-L_d"] = df["high"] - df["low"]
    df["H-PC_d"] = (df["high"] - df["close"].shift(1)).abs()
    df["L-PC_d"] = (df["low"] - df["close"].shift(1)).abs()
    df["TR_d"] = df[["H-L_d", "H-PC_d", "L-PC_d"]].max(axis=1)
    df["atr14_d"] = df["TR_d"].rolling(14).mean()

    # close futuro a H_EMA_DAYS
    df["future_close_ema"] = df["close"].shift(-HORIZON_EMA_BARS)

    k = K_MAP_EMA.get(symbol, 1.0)
    df["thr_ema"] = df["atr14_d"] * k

    df["target_ema50_1d"] = 0
    df.loc[df["future_close_ema"] > df["ema50_d"] + df["thr_ema"], "target_ema50_1d"] = 1
    df.loc[df["future_close_ema"] < df["ema50_d"] - df["thr_ema"], "target_ema50_1d"] = -1

    keep_cols = ["timestamp", "open", "high", "low", "close", "volume", "target_ema50_1d"]
    return df[keep_cols]


def train_for_symbol_ema50_1d(symbol: str):
    csv_path = os.path.join(DATA_DIR, f"{symbol}_1d.csv")
    if not os.path.exists(csv_path):
        print(f"[{symbol}][EMA50_1D] CSV non trovato: {csv_path}")
        return

    print("\n====================================================")
    print(f"  TRAIN EMA50 DAILY (H={H_EMA_DAYS}d): {symbol}")
    print("====================================================\n")

    df_raw = pd.read_csv(csv_path)
    df_lbl = add_target_ema50_1d(df_raw, symbol)

    # feature engineering daily
    df_feat = add_features_daily(df_lbl)

    if "target_ema50_1d" not in df_feat.columns:
        print(f"[{symbol}][EMA50_1D] target_ema50_1d non presente dopo le feature.")
        return

    # rimuovi neutri
    df = df_feat[df_feat["target_ema50_1d"] != 0].copy()
    if len(df) < 1000:
        print(f"[{symbol}][EMA50_1D] pochi dati dopo filtro target!=0: {len(df)}")
        return

    # target binario: +1 -> 1, -1 -> 0
    df["target_bin"] = (df["target_ema50_1d"] == 1).astype(int)

    df = df.sort_values("timestamp")
    y = df["target_bin"]
    X = df.drop(columns=["target_ema50_1d", "target_bin"], errors="ignore")
    if "timestamp" in X.columns:
        X = X.drop(columns=["timestamp"])

    print(f"[{symbol}][EMA50_1D] Dopo cleanup: X={len(X)}, y={len(y)}")

    # split temporale 75/25
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, shuffle=False
    )

    pos = y_train.sum()
    neg = len(y_train) - pos
    if pos == 0 or neg == 0:
        print(f"[{symbol}][EMA50_1D] Una sola classe nel train (pos={pos}, neg={neg}).")
        return

    scale_pos_weight = neg / (pos + 1e-9)
    print(f"[{symbol}][EMA50_1D] pos={pos}, neg={neg}, scale_pos_weight={scale_pos_weight:.2f}")

    # ============================
    # XGBoost
    # ============================
    print(f"[{symbol}][EMA50_1D] Addestro XGBoost…")
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
    print(f"[{symbol}][EMA50_1D] XGB accuracy (thr=0.5): {acc_xgb:.4f}")

    # ============================
    # LightGBM
    # ============================
    print(f"[{symbol}][EMA50_1D] Addestro LightGBM…")
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
    print(f"[{symbol}][EMA50_1D] LGB accuracy (thr=0.5): {acc_lgb:.4f}")

    # ============================
    # Ensemble MAX
    # ============================
    p_final = np.maximum(p_xgb, p_lgb)
    acc_final = accuracy_score(y_test, (p_final > 0.5).astype(int))
    print(f"[{symbol}][EMA50_1D] ENSEMBLE accuracy (thr=0.5): {acc_final:.4f}")

    thr = optimal_threshold(y_test, p_final)
    print(f"[{symbol}][EMA50_1D] Soglia ottimale ensemble: {thr:.3f}")

    # ============================
    # Salvataggio
    # ============================
    joblib.dump(xgb, os.path.join(MODEL_DIR, f"xgb_ema50_1d_{symbol}.pkl"))
    joblib.dump(lgb, os.path.join(MODEL_DIR, f"lgb_ema50_1d_{symbol}.pkl"))

    with open(os.path.join(MODEL_DIR, f"thr_ema50_1d_{symbol}.txt"), "w") as f:
        f.write(str(thr))

    print(f"[{symbol}][EMA50_1D] Modelli EMA50 DAILY salvati in {MODEL_DIR}/")

    print("\nEsempi previsioni EMA50 DAILY (prime 15 righe test):")
    for i in range(min(15, len(y_test))):
        print(f" idx={i} | true={y_test.iloc[i]} | p_final={p_final[i]:.3f}")


def main():
    for sym in SYMBOLS:
        train_for_symbol_ema50_1d(sym)


if __name__ == "__main__":
    main()
