# features_long_1d.py

import numpy as np
import pandas as pd

def add_features_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    df DAILY con almeno:
      - timestamp, open, high, low, close, volume
      - eventuali colonne target_* (es. target_ema50_1d, target_sma200_1d)
    Ritorna df con:
      - timestamp
      - feature numeriche
      - colonne target_* invariate
    """

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")

    # tipi numerici
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ==============================
    # 1) Return log su varie scale
    # ==============================
    df["log_ret_1"] = np.log(df["close"] / df["close"].shift(1))
    df["log_ret_5"] = np.log(df["close"] / df["close"].shift(5))
    df["log_ret_20"] = np.log(df["close"] / df["close"].shift(20))

    # ==============================
    # 2) Volume normalizzato
    # ==============================
    df["vol_mean_20"] = df["volume"].rolling(20).mean()
    df["vol_norm_20"] = df["volume"] / (df["vol_mean_20"] + 1e-9)

    # ==============================
    # 3) Medie mobili (EMA & SMA)
    # ==============================
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()

    df["sma20"] = df["close"].rolling(20).mean()
    df["sma50"] = df["close"].rolling(50).mean()
    df["sma200"] = df["close"].rolling(200).mean()

    # Distanza relativa da medie
    for base in ["ema20", "ema50", "ema200", "sma20", "sma50", "sma200"]:
        df[f"dist_{base}"] = (df["close"] - df[base]) / (df[base] + 1e-9)

    # ==============================
    # 4) ATR daily
    # ==============================
    df["H-L_daily"] = df["high"] - df["low"]
    df["H-PC_daily"] = (df["high"] - df["close"].shift(1)).abs()
    df["L-PC_daily"] = (df["low"] - df["close"].shift(1)).abs()
    df["TR_daily"] = df[["H-L_daily", "H-PC_daily", "L-PC_daily"]].max(axis=1)
    df["atr14_daily"] = df["TR_daily"].rolling(14).mean()

    # ==============================
    # 5) Slope EMA50
    # ==============================
    df["ema50_shift_5"] = df["ema50"].shift(5)
    df["ema50_slope_5"] = (df["ema50"] - df["ema50_shift_5"]) / (df["atr14_daily"] + 1e-9)

    # ==============================
    # 6) RSI 14 daily
    # ==============================
    delta = df["close"].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)

    roll_up = up.ewm(alpha=1/14, adjust=False).mean()
    roll_down = down.ewm(alpha=1/14, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-9)
    df["rsi14"] = 100 - 100 / (1 + rs)

    # ==============================
    # 7) Pulizia colonne inutili
    # ==============================
    # mantieni timestamp + feature + eventuali target_*
    base_keep = ["timestamp"]
    target_cols = [c for c in df.columns if c.startswith("target_")]

    # colonne da buttare (raw OHLC + intermedi per ATR/medie)
    drop_cols = [
        "open", "high", "low", "close", "volume",
        "vol_mean_20",
        "H-L_daily", "H-PC_daily", "L-PC_daily", "TR_daily",
        "ema20", "ema50", "ema200", "ema50_shift_5",
        "sma20", "sma50", "sma200",
    ]

    feature_cols = [
        c for c in df.columns
        if c not in drop_cols and c not in base_keep and c not in target_cols
    ]

    df = df[base_keep + feature_cols + target_cols]

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    return df
