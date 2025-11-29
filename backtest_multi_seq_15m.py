#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import joblib
from datetime import timedelta
from features_15m import add_features_pro

DATA_DIR = "data_1m"
MODEL_DIR = "modelli"

HORIZON = 15
FEE = 0.0005


# ================================================================
# Carica modelli + soglia
# ================================================================
def load_models_and_threshold(symbol):
    xgb = joblib.load(os.path.join(MODEL_DIR, f"xgb_15m_{symbol}.pkl"))
    lgb = joblib.load(os.path.join(MODEL_DIR, f"lgb_15m_{symbol}.pkl"))

    thr_path = os.path.join(MODEL_DIR, f"thr_15m_{symbol}.txt")
    if os.path.exists(thr_path):
        with open(thr_path, "r") as f:
            thr = float(f.read().strip())
    else:
        thr = 0.5

    return xgb, lgb, thr


# ================================================================
# COSTRUISCI FEATURES IDENTICHE AL TRAINING
# ================================================================
def build_features(symbol):
    path = os.path.join(DATA_DIR, f"{symbol}_1m.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    df_raw = pd.read_csv(path)
    df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"])
    df_raw = df_raw.sort_values("timestamp")

    df_feat = add_features_pro(df_raw)

    return df_raw, df_feat


# ================================================================
# ENSEMBLE PREDICTION
# ================================================================
def generate_predictions(df_feat, xgb, lgb):
    df = df_feat.copy()
    feats = [c for c in df.columns if c not in ["target", "timestamp"]]

    px = xgb.predict_proba(df[feats])[:, 1]
    pl = lgb.predict_proba(df[feats])[:, 1]

    df["p_final"] = np.maximum(px, pl)

    return df


# ================================================================
# SIGNAL ENGINE (long/short)
# ================================================================
def generate_signals(df_pred, thr):
    df = df_pred.copy()

    df["signal"] = np.where(df["p_final"] > thr, "LONG", "SHORT")

    return df


# ================================================================
# COSTRUISCI DATAFRAME PER CALCOLO PN
# ================================================================
def build_backtest_df(df_raw, df_sig):
    df = df_sig.copy()

    df_raw = df_raw.set_index("timestamp")
    df = df.set_index("timestamp")

    df["close"] = df_raw["close"]
    df["future"] = df["close"].shift(-HORIZON)

    df["ret_long"]  = (df["future"] / df["close"] - 1) - FEE
    df["ret_short"] = (df["close"] / df["future"] - 1) - FEE

    df = df.dropna()

    return df.reset_index()


# ================================================================
# SEQUENTIAL TRADE ENGINE
# ================================================================
def backtest_sequential(df):
    trades = []
    in_trade = False
    entry_idx = None
    direction = None

    pnl_total = 0
    wins = 0

    for i in range(len(df)):
        row = df.iloc[i]

        if not in_trade:
            direction = row["signal"]
            in_trade = True
            entry_idx = i

        else:
            if i - entry_idx >= HORIZON:

                if direction == "LONG":
                    ret = df.iloc[i]["ret_long"]
                else:
                    ret = df.iloc[i]["ret_short"]

                pnl_total += ret
                wins += int(ret > 0)

                trades.append({
                    "entry_ts": df.iloc[entry_idx]["timestamp"],
                    "exit_ts":  df.iloc[i]["timestamp"],
                    "direction": direction,
                    "return": ret
                })

                in_trade = False

    if trades:
        winrate = wins / len(trades)
        pnl_mean = pnl_total / len(trades)
    else:
        winrate = 0
        pnl_mean = 0

    return {
        "trades": len(trades),
        "winrate": winrate,
        "pnl_total": pnl_total,
        "pnl_mean": pnl_mean,
        "details": trades
    }


# ================================================================
# BACKTEST SINGOLO SIMBOLO
# ================================================================
def backtest_symbol(symbol):
    print("\n=======================================")
    print(f"   BACKTEST DIREZIONALE 15M: {symbol}")
    print("=======================================\n")

    xgb, lgb, thr = load_models_and_threshold(symbol)
    df_raw, df_feat = build_features(symbol)

    df_pred = generate_predictions(df_feat, xgb, lgb)
    df_sig  = generate_signals(df_pred, thr)
    df_bt   = build_backtest_df(df_raw, df_sig)

    res = backtest_sequential(df_bt)

    print(f"[{symbol}] Trades:      {res['trades']}")
    print(f"[{symbol}] Win-rate:    {res['winrate']:.3f}")
    print(f"[{symbol}] pnl totale:  {res['pnl_total']:.4f}")
    print(f"[{symbol}] pnl medio:   {res['pnl_mean']:.5f}")

    return res


# ================================================================
# MAIN
# ================================================================
if __name__ == "__main__":
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "AVAXUSDT", "XRPUSDT"]

    for sym in symbols:
        backtest_symbol(sym)
