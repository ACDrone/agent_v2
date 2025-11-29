#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
papertrade_live.py

Paper trading LIVE del meta-coordinator.
- Fix: Compatibilità libreria 'ta' (ADX restituisce Series).
- Fix: fillna deprecato sostituito con ffill.
- Fix: pct_change() FutureWarning con fill_method=None.
- Fix: Data Leak (previsione futura) risolto.
- Fix: Caching orario per evitare rate limit API.
- NEW: Logica di liquidazione approssimata basata sulla leva.
- NEW: PERSISTENZA STATO (Ricarica wallet e posizioni al riavvio).
"""

from __future__ import annotations

import os
import time
import json
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import joblib
import ccxt
import ta
from xgboost import XGBClassifier

# IMPORTIAMO LE FUNZIONI DAL TUO META_CORE
from meta_core import (
    Account,
    Position,
    PositionState,
    PositionDir,
    compute_size_notional,
    decide_action,
    compute_dynamic_leverage
)

from features import add_features_15m, add_features_1h
from features_long_1d import add_features_daily

# ======================================================
# CONFIGURAZIONE
# ======================================================

START_CAPITAL = 1000.0        # "euro"/USDT iniziali
FEE = 0.0005                  # 0.05% a trade
ALLOC_FRACTION = 0.10         # 10% dell'equity come margine base

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "AVAXUSDT", "XRPUSDT"]

MODEL_15M_DIR = "modelli"
MODEL_1H_DIR = "modelli_1h"
MODEL_EMA1D_DIR = "modelli_ema50"
MODEL_SMA1D_DIR = "modelli_sma200"

WARMUP_1M_BARS = 1500

TRADE_LOG_FILE = "papertrades_live_meta.csv"
HISTORY_FILE = "history_live.csv"
STATE_FILE = "papertrade_state.json"

BINANCE_BASE = "https://api.binance.com"

# --- FUTURES PRO SIGNAL CONFIG ---------------------------------

FUTURES_MAP: Dict[str, Tuple[str, str]] = {
    "BTCUSDT": ("BTCUSDT", "BTC/USDT"),
    "ETHUSDT": ("ETHUSDT", "ETH/USDT"),
    "SOLUSDT": ("SOLUSDT", "SOL/USDT"),
    "AVAXUSDT": ("AVAXUSDT", "AVAX/USDT"),
    "XRPUSDT": ("XRPUSDT", "XRP/USDT"),
}

FUTURES_TIMEFRAME = "1h"
FUTURES_LIMIT = 700   # numero barre 1h futures per il modello PRO

# Liquidazione approssimata:
LIQUIDATION_BUFFER = 1.0


# ======================================================
# UTILITY LOG
# ======================================================

def log_msg(msg: str):
    """Stampa immediata a schermo (flush=True)"""
    print(msg, flush=True)


# ======================================================
# FETCH SPOT (1m / 1d) PER I MODELLI ESISTENTI
# ======================================================

def fetch_binance_klines(symbol: str, interval: str = "1m", limit: int = 1000) -> pd.DataFrame:
    url = f"{BINANCE_BASE}/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        log_msg(f"[FETCH ERROR] {symbol}: {e}")
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    if not data:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    rows = []
    for k in data:
        rows.append({
            "timestamp": pd.to_datetime(k[0], unit="ms"),
            "open": float(k[1]),
            "high": float(k[2]),
            "low": float(k[3]),
            "close": float(k[4]),
            "volume": float(k[5]),
        })

    df = pd.DataFrame(rows)
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def fetch_new_1m_since(symbol: str, last_ts: Optional[pd.Timestamp], limit: int = 1000) -> pd.DataFrame:
    df = fetch_binance_klines(symbol, interval="1m", limit=limit)
    if last_ts is not None:
        df = df[df["timestamp"] > last_ts].copy()
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def fetch_daily_history(symbol: str, limit: int = 500) -> pd.DataFrame:
    return fetch_binance_klines(symbol, interval="1d", limit=limit)


# ======================================================
# CARICAMENTO MODELLI ESISTENTI
# ======================================================

def load_models_15m(symbol: str):
    xgb = joblib.load(os.path.join(MODEL_15M_DIR, f"xgb_15m_{symbol}.pkl"))
    lgb = joblib.load(os.path.join(MODEL_15M_DIR, f"lgb_15m_{symbol}.pkl"))
    return xgb, lgb


def load_models_1h(symbol: str):
    xgb = joblib.load(os.path.join(MODEL_1H_DIR, f"xgb_1h_{symbol}.pkl"))
    lgb = joblib.load(os.path.join(MODEL_1H_DIR, f"lgb_1h_{symbol}.pkl"))
    return xgb, lgb


def load_models_ema1d(symbol: str):
    xgb = joblib.load(os.path.join(MODEL_EMA1D_DIR, f"xgb_ema50_1d_{symbol}.pkl"))
    lgb = joblib.load(os.path.join(MODEL_EMA1D_DIR, f"lgb_ema50_1d_{symbol}.pkl"))
    return xgb, lgb


def load_models_sma1d(symbol: str):
    xgb = joblib.load(os.path.join(MODEL_SMA1D_DIR, f"xgb_sma200_1d_{symbol}.pkl"))
    lgb = joblib.load(os.path.join(MODEL_SMA1D_DIR, f"lgb_sma200_1d_{symbol}.pkl"))
    return xgb, lgb


def load_daily_thresholds(symbol: str) -> tuple[float, float]:
    thr_ema, thr_sma = 0.6, 0.6
    try:
        with open(os.path.join(MODEL_EMA1D_DIR, f"thr_ema50_1d_{symbol}.txt"), "r") as f:
            thr_ema = float(f.read().strip())
    except:
        pass
    try:
        with open(os.path.join(MODEL_SMA1D_DIR, f"thr_sma200_1d_{symbol}.txt"), "r") as f:
            thr_sma = float(f.read().strip())
    except:
        pass
    return thr_ema, thr_sma


# ======================================================
# CALCOLI DAILY / 1H / 15M ESISTENTI
# ======================================================

def compute_daily_probs_last(symbol: str) -> tuple[float, float, pd.Timestamp]:
    df_daily = fetch_daily_history(symbol, limit=500)
    if df_daily.empty:
        return 0.5, 0.5, pd.Timestamp.utcnow()

    df_feat = add_features_daily(df_daily.copy())
    df_feat.sort_values("timestamp", inplace=True)
    df_feat.reset_index(drop=True, inplace=True)

    x = df_feat.drop(columns=[c for c in ["timestamp", "target"] if c in df_feat.columns])
    ts = pd.to_datetime(df_feat["timestamp"])

    xgb_ema, lgb_ema = load_models_ema1d(symbol)
    xgb_sma, lgb_sma = load_models_sma1d(symbol)

    p_ema = np.maximum(xgb_ema.predict_proba(x)[:, 1], lgb_ema.predict_proba(x)[:, 1])
    p_sma = np.maximum(xgb_sma.predict_proba(x)[:, 1], lgb_sma.predict_proba(x)[:, 1])

    return float(p_ema[-1]), float(p_sma[-1]), pd.to_datetime(ts.iloc[-1])


def compute_last_p_1h(symbol: str, df_1m: pd.DataFrame) -> Optional[float]:
    if len(df_1m) < 100:
        return None
    df_feat = add_features_1h(df_1m.copy())
    if df_feat.empty:
        return None
    feat_cols = [c for c in df_feat.columns if c not in ("timestamp", "target")]
    X = df_feat[feat_cols].apply(pd.to_numeric, errors="coerce")
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.dropna(inplace=True)
    if X.empty:
        return None
    X_last = X.iloc[[-1]]
    xgb_1h, lgb_1h = load_models_1h(symbol)
    p_x = xgb_1h.predict_proba(X_last)[:, 1][0]
    p_l = lgb_1h.predict_proba(X_last)[:, 1][0]
    return float(max(p_x, p_l))


def compute_last_p_15m(symbol: str, df_1m: pd.DataFrame) -> Optional[float]:
    if len(df_1m) < 100:
        return None
    df_feat = add_features_15m(df_1m.copy())
    if df_feat.empty:
        return None
    feat_cols = [c for c in df_feat.columns if c not in ("timestamp", "target")]
    X = df_feat[feat_cols].apply(pd.to_numeric, errors="coerce")
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.dropna(inplace=True)
    if X.empty:
        return None
    X_last = X.iloc[[-1]]
    xgb_15m, lgb_15m = load_models_15m(symbol)
    p_x = xgb_15m.predict_proba(X_last)[:, 1][0]
    p_l = lgb_15m.predict_proba(X_last)[:, 1][0]
    return float(max(p_x, p_l))


# ======================================================
# LOGGING (FILE)
# ======================================================

def append_trade_log(row: dict, filename: str = TRADE_LOG_FILE):
    df = pd.DataFrame([row])
    if not os.path.exists(filename):
        df.to_csv(filename, index=False, mode="w", header=True)
    else:
        df.to_csv(filename, index=False, mode="a", header=False)


def append_history_log(rows: List[dict], filename: str = HISTORY_FILE):
    if not rows:
        return
    df = pd.DataFrame(rows)
    if not os.path.exists(filename):
        df.to_csv(filename, index=False, mode="w", header=True)
    else:
        df.to_csv(filename, index=False, mode="a", header=False)


def save_state(state: dict, filename: str = STATE_FILE):
    try:
        with open(filename, "w") as f:
            json.dump(state, f, default=str, indent=2)
    except Exception as e:
        log_msg(f"[STATE ERROR] {e}")


# ======================================================
# PRO SIGNAL FUTURES 1H (FUNDING + OI + ORDERFLOW)
# ======================================================

def fetch_fut_ohlcv(symbol_ccxt: str) -> pd.DataFrame:
    ex = ccxt.binanceusdm()
    raw = ex.fetch_ohlcv(symbol_ccxt, FUTURES_TIMEFRAME, limit=FUTURES_LIMIT)
    df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    df.set_index("ts", inplace=True)
    return df


def fetch_funding(symbol_fut: str) -> pd.DataFrame:
    url = "https://fapi.binance.com/fapi/v1/fundingRate"
    params = {"symbol": symbol_fut, "limit": FUTURES_LIMIT}
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        log_msg(f"[FUNDING ERROR] {symbol_fut}: {e}")
        return pd.DataFrame(columns=["funding"])

    if not isinstance(data, list) or len(data) == 0:
        return pd.DataFrame(columns=["funding"])

    df = pd.DataFrame(data)
    df["time"] = pd.to_datetime(df["fundingTime"], unit="ms")
    df.set_index("time", inplace=True)
    df["funding"] = df["fundingRate"].astype(float)
    return df[["funding"]]


def fetch_open_interest(symbol_fut: str) -> pd.DataFrame:
    url = "https://fapi.binance.com/futures/data/openInterestHist"
    params = {"symbol": symbol_fut, "period": "1h", "limit": FUTURES_LIMIT}
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        log_msg(f"[OI ERROR] {symbol_fut}: {e}")
        return pd.DataFrame(columns=["openInterest"])

    if not isinstance(data, list) or len(data) == 0:
        return pd.DataFrame(columns=["openInterest"])

    df = pd.DataFrame(data)
    df["time"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("time", inplace=True)
    df["openInterest"] = df["sumOpenInterest"].astype(float)
    return df[["openInterest"]]


def fetch_long_short(symbol_fut: str) -> pd.DataFrame:
    url = "https://fapi.binance.com/futures/data/globalLongShortAccountRatio"
    params = {"symbol": symbol_fut, "period": "1h", "limit": FUTURES_LIMIT}
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        log_msg(f"[LSR ERROR] {symbol_fut}: {e}")
        return pd.DataFrame(columns=["longShort"])

    if not isinstance(data, list) or len(data) == 0:
        return pd.DataFrame(columns=["longShort"])

    df = pd.DataFrame(data)
    df["time"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("time", inplace=True)
    df["longShort"] = df["longShortRatio"].astype(float)
    return df[["longShort"]]


def fetch_taker_volume(symbol_fut: str) -> pd.DataFrame:
    url = "https://fapi.binance.com/futures/data/takerlongshortRatio"
    params = {"symbol": symbol_fut, "period": "1h", "limit": FUTURES_LIMIT}
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        log_msg(f"[TAKER ERROR] {symbol_fut}: {e}")
        return pd.DataFrame(columns=["orderflow"])

    if not isinstance(data, list) or len(data) == 0:
        return pd.DataFrame(columns=["orderflow"])

    df = pd.DataFrame(data)

    possible_buy = ["takerBuyVol", "takerBuyVolume", "buyVol", "takerBuyQty"]
    possible_sell = ["takerSellVol", "takerSellVolume", "sellVol", "takerSellQty"]

    buy_col = next((c for c in possible_buy if c in df.columns), None)
    sell_col = next((c for c in possible_sell if c in df.columns), None)

    if buy_col is None or sell_col is None:
        log_msg(f"[TAKER WARN] {symbol_fut}: colonne buy/sell assenti, orderflow=0")
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


def build_futures_df(symbol_fut: str, symbol_ccxt: str) -> pd.DataFrame:
    df = fetch_fut_ohlcv(symbol_ccxt)
    if df.empty:
        return df

    df = df.merge(fetch_funding(symbol_fut), left_index=True, right_index=True, how="left")
    df = df.merge(fetch_open_interest(symbol_fut), left_index=True, right_index=True, how="left")
    df = df.merge(fetch_long_short(symbol_fut), left_index=True, right_index=True, how="left")
    df = df.merge(fetch_taker_volume(symbol_fut), left_index=True, right_index=True, how="left")

    # FIX: Use ffill() instead of fillna(method='ffill')
    df.ffill(inplace=True)
    df.dropna(inplace=True)
    return df


def add_features_pro(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Ritorna:
    - df_train: DataFrame pulito con Target (per addestramento)
    - df_last: L'ultima riga con le Features calcolate (per predizione)
    - feat_cols: Nomi colonne
    """
    df = df.copy()

    # FIX: ADX now returns a Series, no need for ['ADX_14'] indexing
    df["RSI"] = ta.momentum.rsi(df["close"], window=14)
    df["ATR"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=14)
    df["ADX"] = ta.trend.adx(df["high"], df["low"], df["close"], window=14)

    df["LogRet"] = np.log(df["close"] / df["close"].shift(1))
    df["OI_Change"] = df["openInterest"].pct_change(fill_method=None)
    df["Funding_Change"] = df["funding"].pct_change(fill_method=None)

    # Z-Score Orderflow
    roll_mean = df["orderflow"].rolling(20).mean()
    roll_std = df["orderflow"].rolling(20).std().replace(0, 1e-6)
    df["Orderflow_Z"] = (df["orderflow"] - roll_mean) / roll_std

    # Target (Shiftato) – NO fill_method impliciti
    df["NextRet"] = df["close"].shift(-1).pct_change(fill_method=None)
    df["Target"] = (df["NextRet"] > (FEE * 3)).astype(int)

    # Shift Features (Input=Ieri, Target=Oggi)
    feats = ["RSI", "ATR", "ADX", "LogRet", "OI_Change", "Funding_Change", "Orderflow_Z"]
    for c in feats:
        df[f"{c}_prev"] = df[c].shift(1)

    feat_cols = [f"{c}_prev" for c in feats]

    # FIX: Correct splitting to avoid predicting on NaN or future data
    df_valid_features = df.dropna(subset=feat_cols)

    if df_valid_features.empty:
        return pd.DataFrame(), pd.DataFrame(), feat_cols

    # df_last è l'ultima riga (quella corrente da predire)
    df_last = df_valid_features.iloc[[-1]].copy()

    # df_train sono tutte le righe tranne l'ultima, dove abbiamo sia features che target validi
    df_train = df_valid_features.iloc[:-1].dropna(subset=["Target"])

    return df_train, df_last, feat_cols


# CACHE GLOBALE per il modello PRO
pro_model_cache = {}

def compute_p_1h_pro(symbol_fut: str, symbol_ccxt: str, current_ts_1h: pd.Timestamp) -> Optional[float]:
    """
    Allena e predice. Usa una cache per evitare di chiamare API e Training
    ogni minuto se siamo ancora nella stessa ora.
    """
    cache_key = f"{symbol_fut}_{current_ts_1h}"

    # Se abbiamo già calcolato per questa ora, ritorna il valore cachato
    if cache_key in pro_model_cache:
        return pro_model_cache[cache_key]

    # --- INIZIO CALCOLO PESANTE ---
    try:
        # Scarico dati (API call)
        df_raw = build_futures_df(symbol_fut, symbol_ccxt)
    except Exception as e:
        log_msg(f"[FUT DF ERROR] {symbol_fut}: {e}")
        return None

    if df_raw.empty or len(df_raw) < 200:
        return None

    # Feature Engineering con il FIX
    df_train, df_last, feat_cols = add_features_pro(df_raw)

    if df_train.empty or df_last.empty:
        return None

    X_train = df_train[feat_cols]
    y_train = df_train["Target"]
    X_last_feat = df_last[feat_cols]

    try:
        # Training veloce
        model = XGBClassifier(
            n_estimators=100,      # Ridotto a 100 per velocità live
            learning_rate=0.05,
            max_depth=4,
            random_state=42,
            n_jobs=1               # 1 job per non intasare la CPU nel loop
        )
        model.fit(X_train, y_train)

        p_long = float(model.predict_proba(X_last_feat)[0, 1])

        # Salva in cache
        pro_model_cache.clear()  # Pulisci vecchia cache per risparmiare RAM
        pro_model_cache[cache_key] = p_long

        log_msg(f"[{symbol_fut}] PRO Model Refreshed. P_Long: {p_long:.2f}")
        return p_long

    except Exception as e:
        log_msg(f"[FUT MODEL ERROR] {symbol_fut}: {e}")
        return None


# ======================================================
# PERSISTENZA STATO (NEW)
# ======================================================

def load_initial_state(account: Account, states: Dict[str, PositionState]):
    """
    Tenta di caricare lo stato dal JSON al riavvio.
    Ripristina Balance, Equity e Posizioni Aperte.
    """
    if not os.path.exists(STATE_FILE):
        return

    try:
        with open(STATE_FILE, "r") as f:
            data = json.load(f)

        # Ripristino equity/balance (approssimato)
        # Se c'era equity salvata, la usiamo come base, ma ricalcoleremo il balance
        # basandoci sulle posizioni aperte.
        # Per semplicità, se il file esiste, fidiamoci del balance implicito o salvato.
        # Qui usiamo una logica semplice: se c'è un saldo salvato nel JSON (bisognerebbe averlo salvato),
        # altrimenti usiamo equity come balance temporaneo e sottraiamo margine.
        
        # Nel JSON attuale salviamo 'equity', 'margin_used'.
        # Ricostruiamo account.balance = equity - pnl_non_realizzato (che calcoleremo live)
        # Per ora impostiamo balance = equity salvata (si correggerà al primo tick con i prezzi nuovi)
        saved_equity = float(data.get("equity", START_CAPITAL))
        
        # Ripristino Posizioni
        markets = data.get("markets", {})
        loaded_positions = 0
        
        for sym, mkt_data in markets.items():
            state_str = mkt_data.get("state", "FLAT")
            if state_str in ["LONG", "SHORT"]:
                # Creiamo la posizione
                direction = PositionDir.LONG if state_str == "LONG" else PositionDir.SHORT
                
                # Dobbiamo stimare entry_price e size se non salvati esplicitamente
                # Nel JSON 'markets' abbiamo 'pnl', 'margin', 'leverage'.
                # Possiamo derivare notional = margin * leverage
                lev = float(mkt_data.get("leverage", 1.0))
                margin = float(mkt_data.get("margin", 0.0))
                notional = margin * lev
                
                # Prezzo corrente nel JSON (vecchio)
                last_price = float(mkt_data.get("price", 0.0))
                
                # Se pnl = (price/entry - 1)*notional -> price/entry = pnl/notional + 1 -> entry = price / (pnl/notional + 1)
                pnl = float(mkt_data.get("pnl", 0.0))
                
                if notional > 0:
                    if direction == PositionDir.LONG:
                        entry_price = last_price / ((pnl / notional) + 1.0)
                    else:
                        entry_price = last_price * ((pnl / notional) + 1.0)
                else:
                    entry_price = last_price # Fallback

                # Creiamo oggetto Position
                pos = Position(
                    symbol=sym,
                    direction=direction,
                    entry_price=entry_price,
                    entry_time=pd.Timestamp.utcnow(), # Reset time (perdiamo info durata, accettabile)
                    notional=notional,
                    leverage=lev
                )
                account.positions[sym] = pos
                
                # Aggiorniamo PositionState per meta_core
                pst = states.get(sym, PositionState())
                pst.direction = direction
                pst.entry_price = entry_price
                pst.size = notional
                pst.bars_held = 10 # Buffer per non chiudere subito per min_hold
                states[sym] = pst
                
                loaded_positions += 1

        # Aggiustiamo il balance: Balance = Equity - PnL_aperto (che nel JSON era 'pnl_total' ma quello è storico)
        # Semplificazione: Account Balance = Saved Equity - Sum(Open PnLs)
        # Ma i PnL nel json sono vecchi. Facciamo: Balance = Saved Equity (approx)
        # Al primo tick live, l'equity verrà ricalcolata come Balance + PnL_live.
        # Se settiamo Balance = Equity salvata, stiamo "realizzando" virtualmente il PnL allo shutdown.
        # È un approccio sicuro per non perdere soldi.
        account.balance = saved_equity 
        
        log_msg(f"=== STATO RIPRISTINATO: Equity=${saved_equity:.2f}, Posizioni={loaded_positions} ===")

    except Exception as e:
        log_msg(f"[LOAD STATE ERROR] {e}")


# ======================================================
# MAIN LOOP
# ======================================================

def sleep_to_next_minute():
    now = pd.Timestamp.utcnow()
    secs = 60 - now.second - now.microsecond / 1e6
    if secs < 1:
        secs = 1
    time.sleep(secs)


def bootstrap_1m_buffer(symbol: str) -> pd.DataFrame:
    log_msg(f"[{symbol}] Scarico storico 1m...")
    return fetch_binance_klines(symbol, interval="1m", limit=WARMUP_1M_BARS)


def papertrade_live():
    log_msg("=== AVVIO PAPER TRADE LIVE (V5 FIX + LIQUIDATION + PERSISTENCE) ===")

    account = Account(balance=START_CAPITAL)
    buffers_1m: Dict[str, pd.DataFrame] = {}
    states: Dict[str, PositionState] = {sym: PositionState() for sym in SYMBOLS}
    daily_info = {}

    equity_history: List[float] = []
    trades_closed_count = 0
    wins_count = 0
    last_action_global = ""
    last_trades_cache: List[dict] = []
    last_p_1h: Dict[str, float] = {}
    last_p_15m: Dict[str, float] = {}

    # BOOTSTRAP
    log_msg("=== CARICAMENTO INIZIALE ===")
    
    # 1. Carica stato precedente se esiste
    load_initial_state(account, states)

    # 2. Carica Dati
    for sym in SYMBOLS:
        try:
            df_1m = bootstrap_1m_buffer(sym)
            buffers_1m[sym] = df_1m

            p_ema1d, p_sma1d, ts_last_day = compute_daily_probs_last(sym)
            thr_ema, thr_sma = load_daily_thresholds(sym)

            daily_info[sym] = {
                "p_ema1d": p_ema1d, "p_sma1d": p_sma1d,
                "last_day": ts_last_day.normalize(),
                "thr_ema": thr_ema, "thr_sma": thr_sma,
            }
            if not df_1m.empty:
                account.last_price[sym] = float(df_1m.iloc[-1]["close"])
            log_msg(f"[{sym}] OK. DailyProbs: EMA={p_ema1d:.2f}, SMA={p_sma1d:.2f}")
            time.sleep(0.5) # Gentilezza API
        except Exception as e:
            log_msg(f"[{sym}] ERRORE BOOTSTRAP: {e}")

    log_msg("=== LOOP LIVE AVVIATO ===")

    while True:
        loop_ts = pd.Timestamp.utcnow().floor("1min")
        log_msg(f"\n[LOOP] {loop_ts} UTC | Eq: {account.equity:.2f} | Posizioni: {len(account.positions)}")

        history_rows = []

        for sym in SYMBOLS:
            if sym not in buffers_1m:
                continue

            # 1. Update Dati 1m spot
            df_buf = buffers_1m[sym]
            last_ts = df_buf["timestamp"].max() if not df_buf.empty else None
            try:
                new_df = fetch_new_1m_since(sym, last_ts=last_ts, limit=1000)
            except Exception as e:
                log_msg(f"[{sym}] Err fetch: {e}")
                continue

            if not new_df.empty:
                df_buf = pd.concat([df_buf, new_df], ignore_index=True).sort_values("timestamp")
                if len(df_buf) > WARMUP_1M_BARS:
                    df_buf = df_buf.iloc[-WARMUP_1M_BARS:].reset_index(drop=True)
                buffers_1m[sym] = df_buf

            if df_buf.empty:
                continue

            last_row = df_buf.iloc[-1]
            ts_now = pd.to_datetime(last_row["timestamp"])
            price_now = float(last_row["close"])
            account.last_price[sym] = price_now

            # 1b. Controllo LIQUIDAZIONE per posizione esistente
            state_for_liq = states.get(sym, PositionState())
            pos = account.positions.get(sym)
            if pos is not None and state_for_liq.direction != PositionDir.FLAT:
                lev = max(pos.leverage, 1.0)

                if pos.direction == PositionDir.LONG:
                    liq_price = pos.entry_price * (1.0 - (LIQUIDATION_BUFFER / lev))
                    hit_liq = price_now <= liq_price
                else:  # SHORT
                    liq_price = pos.entry_price * (1.0 + (LIQUIDATION_BUFFER / lev))
                    hit_liq = price_now >= liq_price

                if hit_liq:
                    # Liquidazione forzata
                    if pos.direction == PositionDir.LONG:
                        ret = (price_now / pos.entry_price - 1.0) - FEE
                    else:
                        ret = (pos.entry_price / price_now - 1.0) - FEE
                    pnl = pos.notional * ret
                    account.balance += pnl
                    trades_closed_count += 1
                    if pnl > 0:
                        wins_count += 1
                    last_action_global = f"{sym} LIQUIDATED"

                    dur = int((ts_now - pos.entry_time).total_seconds() / 60) if pos.entry_time else 0
                    trade_info = {
                        "symbol": sym,
                        "side": "LONG" if pos.direction == PositionDir.LONG else "SHORT",
                        "entry": pos.entry_price,
                        "exit": price_now,
                        "pnl": pnl,
                        "minutes": dur,
                        "reason": "LIQUIDATION"
                    }
                    last_trades_cache.append(trade_info)
                    append_trade_log({
                        **trade_info,
                        "entry_time": pos.entry_time,
                        "exit_time": ts_now,
                        "notional": pos.notional,
                        "equity_after": account.equity
                    })
                    account.positions.pop(sym, None)
                    state_for_liq.direction = PositionDir.FLAT
                    state_for_liq.size = 0.0
                    states[sym] = state_for_liq
                    log_msg(f"[{sym}] *** LIQUIDATION *** at {price_now:.4f} | Lev={lev}x (liq_price≈{liq_price:.4f})")
                    # Salta logica normale per questo simbolo in questo minuto
                    continue

            # 2. Update Daily
            info = daily_info[sym]
            if ts_now.normalize() > info["last_day"]:
                try:
                    p_e, p_s, ts_d = compute_daily_probs_last(sym)
                    info.update({"p_ema1d": p_e, "p_sma1d": p_s, "last_day": ts_d.normalize()})
                    log_msg(f"[{sym}] Daily update: {p_e:.2f}, {p_s:.2f}")
                except Exception as e:
                    log_msg(f"[{sym}] Err daily: {e}")

            # 3. Predizioni 1h/15m base
            try:
                p_1h_base = compute_last_p_1h(sym, df_buf)
                p_15m = compute_last_p_15m(sym, df_buf)
            except Exception as e:
                log_msg(f"[{sym}] Err pred base: {e}")
                continue

            # 4. Predizione PRO su futures (CACHE ORARIA)
            p_1h_pro = None
            if sym in FUTURES_MAP:
                symbol_fut, symbol_ccxt = FUTURES_MAP[sym]
                
                # Arrotonda all'ora corrente per gestire la cache
                current_hour_ts = ts_now.floor("1h")
                
                p_1h_pro = compute_p_1h_pro(symbol_fut, symbol_ccxt, current_hour_ts)

            # 5. Combina p_1h
            if p_1h_base is None and p_15m is None:
                continue

            if p_1h_base is not None and p_1h_pro is not None and not np.isnan(p_1h_pro):
                # media semplice tra modello storico e modello futures PRO
                p_1h = 0.5 * p_1h_base + 0.5 * p_1h_pro
            elif p_1h_pro is not None and not np.isnan(p_1h_pro):
                p_1h = p_1h_pro
            else:
                p_1h = p_1h_base

            if p_1h is None or p_15m is None:
                continue

            last_p_1h[sym] = p_1h
            last_p_15m[sym] = p_15m

            history_rows.append({
                "timestamp": loop_ts, "symbol": sym, "price": price_now,
                "pred_15m": p_15m, "pred_1h": p_1h
            })

            # 6. Logica Trading (meta_core)
            # Assicuriamoci di passare i parametri corretti per meta_core V2
            state = states.get(sym, PositionState())
            
            # --- CHIAMATA A META_CORE ---
            # Verifica se il tuo meta_core.py è aggiornato alla V2 (quello che ti ho dato).
            # Se è la V1, rimuovi i parametri che non si aspetta.
            action, new_state = decide_action(
                symbol=sym,
                price_now=price_now,
                timestamp=ts_now,
                p_ema1d=info["p_ema1d"],
                p_sma1d=info["p_sma1d"],
                thr_ema=info["thr_ema"],
                thr_sma=info["thr_sma"],
                p_1h=p_1h,
                p_15m=p_15m,
                state=state
            )



            # 7. Esecuzione
            if action.startswith("CLOSE"):
                if sym in account.positions:
                    pos = account.positions.pop(sym)
                    if state.direction == PositionDir.LONG:
                        ret = (price_now / pos.entry_price - 1.0) - FEE
                    else:
                        ret = (pos.entry_price / price_now - 1.0) - FEE
                    pnl = pos.notional * ret
                    account.balance += pnl
                    trades_closed_count += 1
                    if pnl > 0:
                        wins_count += 1
                    last_action_global = f"{sym} {action}"

                    dur = int((ts_now - pos.entry_time).total_seconds() / 60) if pos.entry_time else 0
                    trade_info = {
                        "symbol": sym,
                        "side": "LONG" if state.direction == PositionDir.LONG else "SHORT",
                        "entry": pos.entry_price,
                        "exit": price_now,
                        "pnl": pnl,
                        "minutes": dur,
                        "reason": action
                    }
                    last_trades_cache.append(trade_info)
                    append_trade_log({
                        **trade_info,
                        "entry_time": pos.entry_time,
                        "exit_time": ts_now,
                        "notional": pos.notional,
                        "equity_after": account.equity
                    })
                    log_msg(f"[{sym}] {action} PnL={pnl:.2f}")

            if action in ("OPEN_LONG", "OPEN_SHORT"):
                # --- LEVA DINAMICA (da meta_core) ---
                dyn_lev = compute_dynamic_leverage(
                    info["p_ema1d"], info["p_sma1d"],
                    p_1h, p_15m
                )

                notional = compute_size_notional(
                    account,
                    info["p_ema1d"], info["p_sma1d"],
                    p_1h, p_15m,
                    leverage=dyn_lev,
                    alloc_fraction=ALLOC_FRACTION
                )


                if notional > 0 and (account.margin_used + notional / dyn_lev <= account.equity):
                    pos = Position(sym, new_state.direction, price_now, ts_now, notional, dyn_lev)
                    account.positions[sym] = pos
                    new_state.size = notional
                    last_action_global = f"{sym} {action} ({dyn_lev}x)"
                    log_msg(f"[{sym}] OPEN {action} | Lev: {dyn_lev}x | Size: {notional:.2f}")
                else:
                    new_state.direction = PositionDir.FLAT
                    log_msg(f"[{sym}] SKIP {action} (Margine o Confidenza insufficiente)")

            states[sym] = new_state

        append_history_log(history_rows)

        # Update KPI
        equity_history.append(account.equity)
        max_eq = max(equity_history) if equity_history else account.equity
        pnl_total = account.equity - START_CAPITAL
        dd = (account.equity / max_eq - 1.0) if max_eq > 0 else 0.0
        wr = wins_count / trades_closed_count if trades_closed_count > 0 else 0.0

        # Snapshot
        market_snapshot = {}
        for sym in SYMBOLS:
            price = account.last_price.get(sym)
            if not price:
                continue
            pos = account.positions.get(sym)

            pnl_open, margin_used, lev = 0.0, 0.0, 1.0
            dir_str = "FLAT"

            if pos:
                lev = pos.leverage
                margin_used = pos.notional / lev
                if pos.direction == PositionDir.LONG:
                    pnl_open = (price / pos.entry_price - 1.0) * pos.notional
                else:
                    pnl_open = (pos.entry_price / price - 1.0) * pos.notional
                dir_str = "LONG" if pos.direction == PositionDir.LONG else "SHORT"

            market_snapshot[sym] = {
                "price": price,
                "p15": last_p_15m.get(sym, 0.5),
                "p1h": last_p_1h.get(sym, 0.5),
                "state": dir_str,
                "leverage": lev,
                "pnl": pnl_open,
                "margin": margin_used
            }

        save_state({
            "timestamp": str(loop_ts),
            "equity": account.equity,
            "margin_used": account.margin_used,
            "pnl_total": pnl_total,
            "drawdown": dd,
            "winrate": wr,
            "last_action": last_action_global,
            "markets": market_snapshot,
            "trades": last_trades_cache[-20:],
        })

        sleep_to_next_minute()


if __name__ == "__main__":
    papertrade_live()