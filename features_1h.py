# features_1h.py

import numpy as np
import pandas as pd

def resample_1m_to_1h(df_1m):
    """
    Converte candele 1m in candele 1h.
    df_1m deve avere almeno: timestamp, open, high, low, close, volume
    """
    df = df_1m.copy()

    # timestamp in datetime se non lo è già
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")
    df = df.set_index("timestamp")

    # Usa "1h" per evitare il FutureWarning
    df_1h = df.resample("1h").agg({
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum"
    })

    # rimuovi righe incomplete
    df_1h.dropna(inplace=True)

    # rimetti timestamp come colonna
    df_1h = df_1h.reset_index()  # timestamp torna come colonna

    return df_1h


def add_features_1h(df_1h, fee=0.0005, horizon=1):
    """
    Feature per modello 1h:
    - log_ret_1h
    - atr_1h
    - ema20/50/200 1h + smart200
    - distanze normalizzate
    - rsi_1h

    Target:
    - 1 se close_future > close*(1+fee)
    - 0 altrimenti
    """

    df = df_1h.copy()
    df = df.sort_values("timestamp")

    # ==========================
    # 1. Log return 1h
    # ==========================
    df["log_ret_1h"] = np.log(df["close"] / df["close"].shift(1))

    # ==========================
    # 2. ATR orario
    # ==========================
    df["H-L"]  = df["high"] - df["low"]
    df["H-PC"] = (df["high"] - df["close"].shift(1)).abs()
    df["L-PC"] = (df["low"]  - df["close"].shift(1)).abs()
    df["TR"]   = df[["H-L", "H-PC", "L-PC"]].max(axis=1)
    df["atr_1h"]  = df["TR"].rolling(14).mean()

    # ==========================
    # 3. EMA 20/50/200 + SMART200
    # ==========================
    df["ema20_1h"]  = df["close"].ewm(span=20, adjust=False).mean()
    df["ema50_1h"]  = df["close"].ewm(span=50, adjust=False).mean()
    df["ema200_1h"] = df["close"].ewm(span=200, adjust=False).mean()

    # std su 50 ore per SMART200
    df["std50_1h"] = df["close"].rolling(50).std()

    k = 1.0
    df["smart200_1h"] = df["ema200_1h"] + k * df["std50_1h"]

    # Distanze normalizzate
    df["dist_ema20_1h"]    = (df["close"] - df["ema20_1h"])   / df["ema20_1h"]
    df["dist_ema50_1h"]    = (df["close"] - df["ema50_1h"])   / df["ema50_1h"]
    df["dist_ema200_1h"]   = (df["close"] - df["ema200_1h"])  / df["ema200_1h"]
    df["dist_smart200_1h"] = (df["close"] - df["smart200_1h"]) / df["smart200_1h"]

    # ==========================
    # 4. RSI orario
    # ==========================
    delta = df["close"].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)

    roll_up = up.rolling(14).mean()
    roll_down = down.rolling(14).mean()

    rs = roll_up / (roll_down + 1e-9)
    df["rsi_1h"] = 100 - (100 / (1 + rs))

    # ==========================
    # 5. Target 1h con fee
    # ==========================
    df["future_close"] = df["close"].shift(-horizon)
    df["target"] = (df["future_close"] > df["close"] * (1 + fee)).astype(int)

    # ==========================
    # 6. Pulizia colonne non usabili da XGBoost
    # ==========================
    drop_cols = [
        "open","high","low","close","volume",
        "H-L","H-PC","L-PC","TR","future_close",
        "ema20_1h","ema50_1h","ema200_1h","std50_1h","smart200_1h"
    ]
    for col in drop_cols:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    # importantissimo: niente datetime tra le feature
    if "timestamp" in df.columns:
        df.drop(columns=["timestamp"], inplace=True)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    return df
