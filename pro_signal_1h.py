# pro_signal_1h.py
import ccxt
import pandas as pd
import numpy as np
import ta
import requests
from xgboost import XGBClassifier

TIMEFRAME = "1h"
LIMIT = 700          # stai largo ma non assurdo
FEE = 0.0005         # usato per definire il Target

# ==========================
# 1) WRAPPER DATI FUTURES
# ==========================
def fetch_ohlcv(symbol_ccxt: str):
    ex = ccxt.binanceusdm()
    raw = ex.fetch_ohlcv(symbol_ccxt, TIMEFRAME, limit=LIMIT)
    df = pd.DataFrame(raw, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
    df['ts'] = pd.to_datetime(df['ts'], unit='ms')
    df.set_index('ts', inplace=True)
    return df

def fetch_funding(symbol_fut: str):
    url = "https://fapi.binance.com/fapi/v1/fundingRate"
    params = {"symbol": symbol_fut, "limit": LIMIT}
    r = requests.get(url, params=params).json()
    if not isinstance(r, list) or len(r) == 0:
        return pd.DataFrame(columns=["funding"])
    df = pd.DataFrame(r)
    df["time"] = pd.to_datetime(df["fundingTime"], unit="ms")
    df.set_index("time", inplace=True)
    df["funding"] = df["fundingRate"].astype(float)
    return df[["funding"]]

def fetch_open_interest(symbol_fut: str):
    url = "https://fapi.binance.com/futures/data/openInterestHist"
    params = {"symbol": symbol_fut, "period": "1h", "limit": LIMIT}
    r = requests.get(url, params=params).json()
    if not isinstance(r, list) or len(r) == 0:
        return pd.DataFrame(columns=["openInterest"])
    df = pd.DataFrame(r)
    df["time"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("time", inplace=True)
    df["openInterest"] = df["sumOpenInterest"].astype(float)
    return df[["openInterest"]]

def fetch_long_short(symbol_fut: str):
    url = "https://fapi.binance.com/futures/data/globalLongShortAccountRatio"
    params = {"symbol": symbol_fut, "period": "1h", "limit": LIMIT}
    r = requests.get(url, params=params).json()
    if not isinstance(r, list) or len(r) == 0:
        return pd.DataFrame(columns=["longShort"])
    df = pd.DataFrame(r)
    df["time"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("time", inplace=True)
    df["longShort"] = df["longShortRatio"].astype(float)
    return df[["longShort"]]

def fetch_taker_volume(symbol_fut: str):
    url = "https://fapi.binance.com/futures/data/takerlongshortRatio"
    params = {"symbol": symbol_fut, "period": "1h", "limit": LIMIT}
    r = requests.get(url, params=params).json()
    if not isinstance(r, list) or len(r) == 0:
        return pd.DataFrame(columns=["orderflow"])

    df = pd.DataFrame(r)
    possible_buy  = ["takerBuyVol", "takerBuyVolume", "buyVol", "takerBuyQty"]
    possible_sell = ["takerSellVol", "takerSellVolume", "sellVol", "takerSellQty"]
    buy_col  = next((c for c in possible_buy  if c in df.columns), None)
    sell_col = next((c for c in possible_sell if c in df.columns), None)

    if buy_col is None or sell_col is None:
        df["orderflow"] = 0.0
    else:
        df[buy_col] = df[buy_col].astype(float)
        df[sell_col] = df[sell_col].astype(float)
        df["orderflow"] = df[buy_col] - df[sell_col]

    if "timestamp" in df.columns:
        df["time"] = pd.to_datetime(df["timestamp"], unit="ms")
    else:
        df["time"] = pd.date_range(end=pd.Timestamp.utcnow(), periods=len(df), freq="H")

    df.set_index("time", inplace=True)
    return df[["orderflow"]]

def build_raw_df(symbol_fut: str, symbol_ccxt: str) -> pd.DataFrame:
    df = fetch_ohlcv(symbol_ccxt)
    df = df.merge(fetch_funding(symbol_fut), left_index=True, right_index=True, how="left")
    df = df.merge(fetch_open_interest(symbol_fut), left_index=True, right_index=True, how="left")
    df = df.merge(fetch_long_short(symbol_fut), left_index=True, right_index=True, how="left")
    df = df.merge(fetch_taker_volume(symbol_fut), left_index=True, right_index=True, how="left")
    df.fillna(method="ffill", inplace=True)
    df.dropna(inplace=True)
    return df

# ==========================
# 2) FEATURE + MODELLO
# ==========================
def add_features(df: pd.DataFrame):
    df = df.copy()

    df["RSI"] = ta.momentum.rsi(df["close"], window=14)
    df["ATR"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=14)
    df["ADX"] = ta.trend.adx(df["high"], df["low"], df["close"], window=14)
    df["LogRet"] = np.log(df["close"] / df["close"].shift(1))
    df["OI_Change"] = df["openInterest"].pct_change()
    df["Funding_Change"] = df["funding"].pct_change()
    df["Orderflow_Z"] = (df["orderflow"] - df["orderflow"].rolling(20).mean()) / df["orderflow"].rolling(20).std()

    df["NextRet"] = df["close"].shift(-1).pct_change()
    df["Target"] = (df["NextRet"] > (FEE * 3)).astype(int)

    feats = ["RSI", "ATR", "ADX", "LogRet", "OI_Change", "Funding_Change", "Orderflow_Z"]
    for c in feats:
        df[f"{c}_prev"] = df[c].shift(1)

    df.dropna(inplace=True)
    feat_cols = [f"{c}_prev" for c in feats]
    return df, feat_cols

def compute_p_1h_pro(symbol_fut: str, symbol_ccxt: str) -> float:
    """
    Output: probabilità (0–1) che la prossima candela 1h sia LONG “utile”.
    Pensato per essere chiamato da paper_trading/meta_core.
    """
    df_raw = build_raw_df(symbol_fut, symbol_ccxt)
    if df_raw.empty or len(df_raw) < 300:
        return 0.5

    df, feat_cols = add_features(df_raw)

    split = int(len(df) * 0.80)
    X_train = df[feat_cols].iloc[:split]
    y_train = df["Target"].iloc[:split]
    X_last  = df[feat_cols].iloc[[-1]]

    model = XGBClassifier(
        n_estimators=600,
        learning_rate=0.01,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    p_long = float(model.predict_proba(X_last)[0, 1])
    return p_long
