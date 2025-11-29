#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd


def add_features_pro(df, fee=0.0005, horizon=15, k=0.8):
    """
    Feature engineering avanzato + target direzionale ATR-based su candele 1m.

    target:
      +1  = breakout LONG forte (future > close + ATR*k)
      -1  = breakout SHORT forte (future < close - ATR*k)
       0  = niente movimento significativo (verrà filtrato nel training)

    horizon è espresso in barre da 1m:
      - 15  -> modello 15 minuti
      - 60  -> modello 1h
      - 180 -> modello "EMA50" (~3h)
      - 360 -> modello "SMA200" (~6h)

    k viene passato dal train (K_MAP per simbolo).
    In inferenza/backtest il target viene ignorato, usiamo solo le feature.
    """

    df = df.copy()

    # ------------------------------------------------------
    # Timestamp e ordinamento
    # ------------------------------------------------------
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp")

    # Cast numerico OHLCV
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ------------------------------------------------------
    # LOG RETURN
    # ------------------------------------------------------
    df["log_ret"] = np.log(df["close"] / df["close"].shift(1))

    # ------------------------------------------------------
    # ATR (Average True Range)
    # ------------------------------------------------------
    df["H-L"] = df["high"] - df["low"]
    df["H-PC"] = (df["high"] - df["close"].shift(1)).abs()
    df["L-PC"] = (df["low"] - df["close"].shift(1)).abs()
    df["TR"] = df[["H-L", "H-PC", "L-PC"]].max(axis=1)
    df["atr"] = df["TR"].rolling(14).mean()

    # ------------------------------------------------------
    # DISTANZA DALLE EMA 20/50/100
    # ------------------------------------------------------
    for span in [20, 50, 100]:
        ema = df["close"].ewm(span=span, adjust=False).mean()
        df[f"dist_ema{span}"] = (df["close"] - ema) / ema

    # ------------------------------------------------------
    # SMA200 (trend molto lento) + distanza
    # ------------------------------------------------------
    df["sma200"] = df["close"].rolling(200).mean()
    df["dist_sma200"] = (df["close"] - df["sma200"]) / (df["sma200"] + 1e-9)

    # ------------------------------------------------------
    # RSI (14)
    # ------------------------------------------------------
    delta = df["close"].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)

    roll_up = up.rolling(14).mean()
    roll_down = down.rolling(14).mean()
    rs = roll_up / (roll_down + 1e-9)
    df["rsi"] = 100 - (100 / (1 + rs))

    # ------------------------------------------------------
    # MOMENTUM CLUSTER (1–3 barre precedenti)
    # ------------------------------------------------------
    df["ret1"] = df["log_ret"].shift(1)
    df["ret2"] = df["log_ret"].shift(2)
    df["ret3"] = df["log_ret"].shift(3)
    df["ret_sum3"] = df["ret1"] + df["ret2"] + df["ret3"]

    # ------------------------------------------------------
    # SQUEEZE DI VOLATILITÀ
    # ------------------------------------------------------
    df["atr_mean50"] = df["atr"].rolling(50).mean()
    df["squeeze"] = df["atr"] / (df["atr_mean50"] + 1e-9)

    # ------------------------------------------------------
    # RANGE COMPRESSION
    # ------------------------------------------------------
    df["range"] = df["high"] - df["low"]
    df["range_mean50"] = df["range"].rolling(50).mean()
    df["range_compression"] = df["range"] / (df["range_mean50"] + 1e-9)

    # ------------------------------------------------------
    # REGIME DI VOLATILITÀ
    # ------------------------------------------------------
    df["vol_regime"] = (df["atr"] / df["close"]).rolling(30).mean()

    # ------------------------------------------------------
    # FORZA DEL TREND
    # ------------------------------------------------------
    df["trend_strength"] = df["dist_ema20"] - df["dist_ema50"]

    # ------------------------------------------------------
    # TARGET DIREZIONALE ATR * k
    # ------------------------------------------------------
    df["future"] = df["close"].shift(-horizon)
    df["thr"] = df["atr"] * k

    df["target"] = 0
    df.loc[df["future"] > df["close"] + df["thr"], "target"] = 1      # LONG forte
    df.loc[df["future"] < df["close"] - df["thr"], "target"] = -1     # SHORT forte

    # ------------------------------------------------------
    # DROP COLONNE NON USATE COME FEATURE
    # (teniamo: timestamp, log_ret, atr, dist_ema*, sma200, dist_sma200,
    #  rsi, ret*, squeeze, range_compression, vol_regime, trend_strength, target)
    # ------------------------------------------------------
    drop_cols = [
        "open", "high", "low", "close", "volume",
        "H-L", "H-PC", "L-PC", "TR",
        "atr_mean50", "range_mean50", "range",
        "future", "thr"
    ]

    for c in drop_cols:
        if c in df.columns:
            df.drop(columns=[c], inplace=True)

    # ------------------------------------------------------
    # CLEANUP
    # ------------------------------------------------------
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    return df


# ======================================================================
# WRAPPER PER ORIZZONTI SPECIFICI (stessa logica, nomi più espliciti)
# ======================================================================

def add_features_15m(df, k=0.8):
    """Wrapper per modello 15 minuti (horizon=15)."""
    return add_features_pro(df, horizon=15, k=k)


def add_features_1h(df, k=0.8):
    """Wrapper per modello 1 ora (horizon=60)."""
    return add_features_pro(df, horizon=60, k=k)

