#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_curve, accuracy_score
from sklearn.model_selection import train_test_split

# Cartelle e simboli
DATA_DIR = "data_1m"
MODEL_DIR = "modelli_k_sweep"
os.makedirs(MODEL_DIR, exist_ok=True)

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "AVAXUSDT", "XRPUSDT"]

# Valori di k da testare
K_VALUES = [0.6, 0.8, 1.0, 1.2, 1.5]


# ============================================================
# Soglia ottimale via Youden
# ============================================================
def optimal_threshold(y_true, y_prob):
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    j = tpr - fpr
    best = np.argmax(j)
    return thr[best]


# ============================================================
# Costruisce features + target per un dato k (partendo dal CSV raw)
# ============================================================
def build_features_k(df_raw, k, horizon=15):
    df = df_raw.copy()

    # timestamp + ordinamento
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp")

    # tipi numerici
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # log return
    df["log_ret"] = np.log(df["close"] / df["close"].shift(1))

    # ATR
    df["H-L"] = df["high"] - df["low"]
    df["H-PC"] = (df["high"] - df["close"].shift(1)).abs()
    df["L-PC"] = (df["low"] - df["close"].shift(1)).abs()
    df["TR"] = df[["H-L", "H-PC", "L-PC"]].max(axis=1)
    df["atr"] = df["TR"].rolling(14).mean()

    # EMA distances
    for span in [20, 50, 100]:
        ema = df["close"].ewm(span=span, adjust=False).mean()
        df[f"dist_ema{span}"] = (df["close"] - ema) / ema

    # RSI
    delta = df["close"].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.rolling(14).mean()
    roll_down = down.rolling(14).mean()
    rs = roll_up / (roll_down + 1e-9)
    df["rsi"] = 100 - (100 / (1 + rs))

    # momentum cluster
    df["ret1"] = df["log_ret"].shift(1)
    df["ret2"] = df["log_ret"].shift(2)
    df["ret3"] = df["log_ret"].shift(3)
    df["ret_sum3"] = df["ret1"] + df["ret2"] + df["ret3"]

    # squeeze
    df["atr_mean50"] = df["atr"].rolling(50).mean()
    df["squeeze"] = df["atr"] / (df["atr_mean50"] + 1e-9)

    # range compression
    df["range"] = df["high"] - df["low"]
    df["range_mean50"] = df["range"].rolling(50).mean()
    df["range_compression"] = df["range"] / (df["range_mean50"] + 1e-9)

    # regime di volatilità
    df["vol_regime"] = (df["atr"] / df["close"]).rolling(30).mean()

    # forza del trend
    df["trend_strength"] = df["dist_ema20"] - df["dist_ema50"]

    # filtro orario 12–21 UTC come nel training
    if "timestamp" in df.columns:
        df["hour"] = df["timestamp"].dt.hour
        df = df[df["hour"].between(12, 21)]

    # target direzionale basato su ATR * k
    df["future"] = df["close"].shift(-horizon)
    df["thr"] = df["atr"] * k

    df["target"] = 0
    df.loc[df["future"] > df["close"] + df["thr"], "target"] = 1
    df.loc[df["future"] < df["close"] - df["thr"], "target"] = -1

    # tieni solo i movimenti forti
    df = df[df["target"] != 0]

    # mappatura binaria: +1 → 1 (LONG), -1 → 0 (SHORT)
    df["target_bin"] = (df["target"] == 1).astype(int)

    # drop colonne non usate come feature
    drop_cols = [
        "open", "high", "low", "close", "volume",
        "H-L", "H-PC", "L-PC", "TR",
        "atr_mean50", "range_mean50",
        "range", "future", "thr", "target",
        "timestamp", "hour"
    ]
    for c in drop_cols:
        if c in df.columns:
            df.drop(columns=[c], inplace=True)

    # pulizia
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    return df


# ============================================================
# Allena + valuta per un singolo k
# ============================================================
def eval_k(symbol, k):
    print(f"  → Testo k = {k}")

    csv = os.path.join(DATA_DIR, f"{symbol}_1m.csv")
    if not os.path.exists(csv):
        print(f"     [SKIP] CSV non trovato: {csv}")
        return None

    df_raw = pd.read_csv(csv)
    df = build_features_k(df_raw, k)

    if len(df) < 2000:
        print(f"     [SKIP] pochi esempi: {len(df)}")
        return None

    y = df["target_bin"]
    X = df.drop(columns=["target_bin"])

    # split temporale
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, shuffle=False
    )

    pos = y_train.sum()
    neg = len(y_train) - pos
    if pos == 0 or neg == 0:
        print(f"     [SKIP] classe singola: pos={pos}, neg={neg}")
        return None

    scale_pos_weight = neg / (pos + 1e-9)

    # XGBoost
    xgb = XGBClassifier(
        n_estimators=500,
        max_depth=7,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.9,
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        objective="binary:logistic",
        eval_metric="logloss"
    )
    xgb.fit(X_train, y_train)
    p_xgb = xgb.predict_proba(X_test)[:, 1]

    # LightGBM
    lgb = LGBMClassifier(
        n_estimators=600,
        learning_rate=0.03,
        max_depth=7,
        subsample=0.9,
        colsample_bytree=0.9,
        is_unbalance=True
    )
    lgb.fit(X_train, y_train)
    p_lgb = lgb.predict_proba(X_test)[:, 1]

    # ensemble max()
    p_final = np.maximum(p_xgb, p_lgb)
    pred_final = (p_final > 0.5).astype(int)

    acc = accuracy_score(y_test, pred_final)
    thr_opt = optimal_threshold(y_test, p_final)

    return {
        "k": k,
        "samples": int(len(df)),
        "acc": float(acc),
        "thr": float(thr_opt)
    }


# ============================================================
# Sweep su tutti i k per un simbolo
# ============================================================
def sweep_symbol(symbol):
    print("\n=======================================")
    print(f"K-SWEEP PER {symbol}")
    print("=======================================")

    results = []
    for k in K_VALUES:
        r = eval_k(symbol, k)
        if r is not None:
            results.append(r)

    if not results:
        print(f"[{symbol}] Nessun risultato valido per i K testati.")
        return

    best = sorted(results, key=lambda x: x["acc"], reverse=True)[0]

    print(f"\nMIGLIOR K PER {symbol}: {best['k']}")
    print(f"   Accuracy ensemble: {best['acc']:.3f}")
    print(f"   Soglia ottimale:   {best['thr']:.3f}")
    print(f"   N esempi:          {best['samples']}")

    out_path = os.path.join(MODEL_DIR, f"k_sweep_{symbol}.csv")
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"[{symbol}] Risultati salvati in: {out_path}")

    return best


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    for sym in SYMBOLS:
        sweep_symbol(sym)
