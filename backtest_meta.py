#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
backtest_meta.py

Backtest del meta-coordinator che combina:
- modelli daily EMA50_1D + SMA200_1D (regime macro, da data_1d)
- modello direzionale 1H (bias operativo, orizzonte 60m su dati 1m)
- modello direzionale 15M (timing, orizzonte 15m su dati 1m)

Usa:
  - data_1m/<SYMBOL>_1m.csv
  - data_1d/<SYMBOL>_1d.csv

Dipendenze:
  - features.py            (add_features_15m, add_features_1h)
  - features_long_1d.py    (add_features_daily)
  - modelli:
      * 15m      → modelli/xgb_15m_<SYMBOL>.pkl,  modelli/lgb_15m_<SYMBOL>.pkl
      * 1h       → modelli_1h/xgb_1h_<SYMBOL>.pkl, modelli_1h/lgb_1h_<SYMBOL>.pkl
      * EMA50_1D → modelli_ema50/xgb_ema50_1d_<SYMBOL>.pkl,
                   modelli_ema50/lgb_ema50_1d_<SYMBOL>.pkl,
                   modelli_ema50/thr_ema50_1d_<SYMBOL>.txt
      * SMA200_1D→ modelli_sma200/xgb_sma200_1d_<SYMBOL>.pkl,
                   modelli_sma200/lgb_sma200_1d_<SYMBOL>.pkl,
                   modelli_sma200/thr_sma200_1d_<SYMBOL>.txt
"""

import os
from enum import Enum
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import joblib

# ==============================
# CONFIG
# ==============================

DATA_1M_DIR = "data_1m"
DATA_1D_DIR = "data_1d"

MODEL_15M_DIR = "modelli"
MODEL_1H_DIR = "modelli_1h"
MODEL_EMA1D_DIR = "modelli_ema50"
MODEL_SMA1D_DIR = "modelli_sma200"

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "AVAXUSDT", "XRPUSDT"]

FEE = 0.0005  # 0.05% per trade (ingresso+uscita, approssimato)

# Soglie 1H HIGH-CONFIDENCE (tarate dai tuoi backtest 1h)
THR_HI_1H = {
    "BTCUSDT": 0.80,
    "ETHUSDT": 0.60,
    "SOLUSDT": 0.65,
    "AVAXUSDT": 0.60,
    "XRPUSDT": 0.60,
}
THR_LO_1H = {
    "BTCUSDT": 0.20,
    "ETHUSDT": 0.40,
    "SOLUSDT": 0.35,
    "AVAXUSDT": 0.40,
    "XRPUSDT": 0.40,
}

THR_HI_15M = {
    "BTCUSDT": 0.80,
    "ETHUSDT": 0.70,
    "SOLUSDT": 0.72,
    "AVAXUSDT": 0.72,
    "XRPUSDT": 0.72,
}
THR_LO_15M = {
    "BTCUSDT": 0.20,
    "ETHUSDT": 0.30,
    "SOLUSDT": 0.28,
    "AVAXUSDT": 0.28,
    "XRPUSDT": 0.28,
}


# ==============================
# IMPORT FEATURE FUNCTIONS
# ==============================
# Se il file si chiama diversamente, ad es. features_15m.py,
# cambia questa riga di import di conseguenza.
from features import add_features_15m, add_features_1h
from features_long_1d import add_features_daily

# ==============================
# ENUM, STATO, POLICY
# ==============================

class Regime(Enum):
    BULL = 1
    BEAR = -1
    NEUTRAL = 0

class Bias1H(Enum):
    LONG = 1
    SHORT = -1
    NEUTRAL = 0

class Signal15M(Enum):
    LONG = 1
    SHORT = -1
    NONE = 0

class PositionDir(Enum):
    LONG = 1
    SHORT = -1
    FLAT = 0

@dataclass
class PositionState:
    direction: PositionDir = PositionDir.FLAT
    entry_price: float = 0.0
    entry_time: Optional[pd.Timestamp] = None
    bars_held: int = 0
    size: float = 0.0


def load_daily_thresholds(symbol: str) -> tuple[float, float]:
    """Legge le soglie ROC ottimali salvate per EMA50_1D e SMA200_1D."""
    thr_ema = 0.6
    thr_sma = 0.6

    path_ema = os.path.join(MODEL_EMA1D_DIR, f"thr_ema50_1d_{symbol}.txt")
    if os.path.exists(path_ema):
        try:
            with open(path_ema, "r") as f:
                thr_ema = float(f.read().strip())
        except Exception:
            pass

    path_sma = os.path.join(MODEL_SMA1D_DIR, f"thr_sma200_1d_{symbol}.txt")
    if os.path.exists(path_sma):
        try:
            with open(path_sma, "r") as f:
                thr_sma = float(f.read().strip())
        except Exception:
            pass

    return thr_ema, thr_sma


def compute_regime(symbol: str, p_ema1d: float, p_sma1d: float,
                   thr_ema: float, thr_sma: float) -> Regime:
    bull = (p_ema1d >= thr_ema) and (p_sma1d >= thr_sma)
    bear = (p_ema1d <= 1.0 - thr_ema) and (p_sma1d <= 1.0 - thr_sma)

    if bull and not bear:
        return Regime.BULL
    elif bear and not bull:
        return Regime.BEAR
    else:
        return Regime.NEUTRAL


def compute_bias_1h(symbol: str, p_1h: float) -> Bias1H:
    hi = THR_HI_1H[symbol]
    lo = THR_LO_1H[symbol]
    if p_1h >= hi:
        return Bias1H.LONG
    elif p_1h <= lo:
        return Bias1H.SHORT
    else:
        return Bias1H.NEUTRAL


def compute_signal_15m(symbol: str, p_15m: float) -> Signal15M:
    hi = THR_HI_15M[symbol]
    lo = THR_LO_15M[symbol]
    if p_15m >= hi:
        return Signal15M.LONG
    elif p_15m <= lo:
        return Signal15M.SHORT
    else:
        return Signal15M.NONE


def compute_size(symbol: str,
                 p_ema1d: float,
                 p_sma1d: float,
                 p_1h: float,
                 p_15m: float,
                 base_size: float = 1.0,
                 max_mult: float = 3.0) -> float:
    """Size dinamica in base alla confidenza combinata."""
    conf_regime = abs(p_ema1d - 0.5) + abs(p_sma1d - 0.5)
    conf_1h     = abs(p_1h - 0.5)
    conf_15m    = abs(p_15m - 0.5)

    w1, w2, w3 = 0.5, 0.3, 0.2
    conf_total = w1 * conf_regime + w2 * conf_1h + w3 * conf_15m

    mult = 1.0 + 2.0 * conf_total
    mult = max(0.5, min(mult, max_mult))
    return base_size * mult


def decide_action(symbol: str,
                  price_now: float,
                  p_ema1d: float,
                  p_sma1d: float,
                  thr_ema: float,
                  thr_sma: float,
                  p_1h: float,
                  p_15m: float,
                  state: PositionState) -> tuple[str, PositionState]:
    """
    Ritorna (azione, nuovo_state)
    azione ∈ {"OPEN_LONG", "OPEN_SHORT", "CLOSE", "HOLD", "NOOP"}
    """

    regime = compute_regime(symbol, p_ema1d, p_sma1d, thr_ema, thr_sma)
    bias_1h = compute_bias_1h(symbol, p_1h)
    sig_15m = compute_signal_15m(symbol, p_15m)

    new_state = PositionState(**vars(state))  # copia superficiale

    if new_state.direction != PositionDir.FLAT:
        new_state.bars_held += 1

    MIN_HOLD_BARS = 20
  # minimo numero di step 1m prima di poter chiudere

    # NESSUNA POSIZIONE APERTA → valuta apertura
    if new_state.direction == PositionDir.FLAT:
        # LONG solo se tutti e tre allineati long
        if (regime == Regime.BULL) and (bias_1h == Bias1H.LONG) and (sig_15m == Signal15M.LONG):
            size = compute_size(symbol, p_ema1d, p_sma1d, p_1h, p_15m)
            new_state.direction = PositionDir.LONG
            new_state.entry_price = price_now
            new_state.entry_time = None
            new_state.bars_held = 0
            new_state.size = size
            return "OPEN_LONG", new_state

        # SHORT solo se tutti e tre allineati short
        if (regime == Regime.BEAR) and (bias_1h == Bias1H.SHORT) and (sig_15m == Signal15M.SHORT):
            size = compute_size(symbol, p_ema1d, p_sma1d, p_1h, p_15m)
            new_state.direction = PositionDir.SHORT
            new_state.entry_price = price_now
            new_state.entry_time = None
            new_state.bars_held = 0
            new_state.size = size
            return "OPEN_SHORT", new_state

        return "NOOP", new_state

    # POSIZIONE APERTA → gestione
    if new_state.bars_held < MIN_HOLD_BARS:
        return "HOLD", new_state

    if new_state.direction == PositionDir.LONG:
        if bias_1h != Bias1H.LONG:
            new_state.direction = PositionDir.FLAT
            return "CLOSE", new_state
        if regime == Regime.BEAR:
            new_state.direction = PositionDir.FLAT
            return "CLOSE", new_state
        if sig_15m == Signal15M.SHORT:
            new_state.direction = PositionDir.FLAT
            return "CLOSE", new_state
        return "HOLD", new_state

    if new_state.direction == PositionDir.SHORT:
        if bias_1h != Bias1H.SHORT:
            new_state.direction = PositionDir.FLAT
            return "CLOSE", new_state
        if regime == Regime.BULL:
            new_state.direction = PositionDir.FLAT
            return "CLOSE", new_state
        if sig_15m == Signal15M.LONG:
            new_state.direction = PositionDir.FLAT
            return "CLOSE", new_state
        return "HOLD", new_state

    return "NOOP", new_state


# ==============================
# CARICAMENTO MODELLI
# ==============================

def load_models_15m(symbol: str):
    xgb_path = os.path.join(MODEL_15M_DIR, f"xgb_15m_{symbol}.pkl")
    lgb_path = os.path.join(MODEL_15M_DIR, f"lgb_15m_{symbol}.pkl")
    if not (os.path.exists(xgb_path) and os.path.exists(lgb_path)):
        raise FileNotFoundError(f"Modelli 15m mancanti per {symbol}")
    xgb = joblib.load(xgb_path)
    lgb = joblib.load(lgb_path)
    return xgb, lgb


def load_models_1h(symbol: str):
    xgb_path = os.path.join(MODEL_1H_DIR, f"xgb_1h_{symbol}.pkl")
    lgb_path = os.path.join(MODEL_1H_DIR, f"lgb_1h_{symbol}.pkl")
    if not (os.path.exists(xgb_path) and os.path.exists(lgb_path)):
        raise FileNotFoundError(f"Modelli 1h mancanti per {symbol}")
    xgb = joblib.load(xgb_path)
    lgb = joblib.load(lgb_path)
    return xgb, lgb


def load_models_ema1d(symbol: str):
    xgb_path = os.path.join(MODEL_EMA1D_DIR, f"xgb_ema50_1d_{symbol}.pkl")
    lgb_path = os.path.join(MODEL_EMA1D_DIR, f"lgb_ema50_1d_{symbol}.pkl")
    if not (os.path.exists(xgb_path) and os.path.exists(lgb_path)):
        raise FileNotFoundError(f"Modelli EMA50_1D mancanti per {symbol}")
    xgb = joblib.load(xgb_path)
    lgb = joblib.load(lgb_path)
    return xgb, lgb


def load_models_sma1d(symbol: str):
    xgb_path = os.path.join(MODEL_SMA1D_DIR, f"xgb_sma200_1d_{symbol}.pkl")
    lgb_path = os.path.join(MODEL_SMA1D_DIR, f"lgb_sma200_1d_{symbol}.pkl")
    if not (os.path.exists(xgb_path) and os.path.exists(lgb_path)):
        raise FileNotFoundError(f"Modelli SMA200_1D mancanti per {symbol}")
    xgb = joblib.load(xgb_path)
    lgb = joblib.load(lgb_path)
    return xgb, lgb


# ==============================
# PRECOMPUTO PROB
# ==============================

def compute_daily_probs(symbol: str) -> pd.DataFrame:
    """
    Ritorna df con:
      timestamp (daily),
      p_ema1d,
      p_sma1d
    """
    csv_path = os.path.join(DATA_1D_DIR, f"{symbol}_1d.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV 1d mancante: {csv_path}")

    df_raw = pd.read_csv(csv_path)
    df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"])
    df_raw = df_raw.sort_values("timestamp")

    df_feat = add_features_daily(df_raw)

    if "timestamp" in df_feat.columns:
        ts = pd.to_datetime(df_feat["timestamp"])
        X = df_feat.drop(columns=["timestamp"])
    else:
        raise ValueError("add_features_daily deve mantenere 'timestamp'.")

    xgb_ema, lgb_ema = load_models_ema1d(symbol)
    xgb_sma, lgb_sma = load_models_sma1d(symbol)

    p_ema = np.maximum(
        xgb_ema.predict_proba(X)[:, 1],
        lgb_ema.predict_proba(X)[:, 1],
    )
    p_sma = np.maximum(
        xgb_sma.predict_proba(X)[:, 1],
        lgb_sma.predict_proba(X)[:, 1],
    )

    df_out = pd.DataFrame({
        "timestamp": ts.values,
        "p_ema1d": p_ema,
        "p_sma1d": p_sma,
    }).sort_values("timestamp")

    return df_out


def compute_1h_probs(symbol: str, df_1m: pd.DataFrame) -> pd.DataFrame:
    """
    Usa add_features_1h(df_1m) → feature su 1m con horizon=60.
    Ritorna df con:
      timestamp (1m, per le righe con feature valide),
      p_1h
    """
    df_feat = add_features_1h(df_1m)  # horizon=60 dentro features.py
    if "timestamp" not in df_feat.columns:
        raise ValueError("add_features_1h deve mantenere 'timestamp'.")

    ts = pd.to_datetime(df_feat["timestamp"])

    cols = [c for c in df_feat.columns if c not in ("timestamp", "target")]
    X = df_feat[cols]

    xgb_1h, lgb_1h = load_models_1h(symbol)
    p_1h = np.maximum(
        xgb_1h.predict_proba(X)[:, 1],
        lgb_1h.predict_proba(X)[:, 1],
    )

    df_out = pd.DataFrame({
        "timestamp": ts.values,
        "p_1h": p_1h,
    }).sort_values("timestamp")

    return df_out


def compute_15m_probs(symbol: str, df_1m: pd.DataFrame) -> pd.DataFrame:
    """
    Usa add_features_15m(df_1m) → feature su 1m con horizon=15.
    Ritorna df con:
      timestamp (1m, per le righe con feature valide),
      p_15m
    """
    df_feat = add_features_15m(df_1m)
    if "timestamp" not in df_feat.columns:
        raise ValueError("add_features_15m deve mantenere 'timestamp'.")

    ts = pd.to_datetime(df_feat["timestamp"])
    cols = [c for c in df_feat.columns if c not in ("timestamp", "target")]
    X = df_feat[cols]

    xgb_15m, lgb_15m = load_models_15m(symbol)
    p_15m = np.maximum(
        xgb_15m.predict_proba(X)[:, 1],
        lgb_15m.predict_proba(X)[:, 1],
    )

    df_out = pd.DataFrame({
        "timestamp": ts.values,
        "p_15m": p_15m,
    }).sort_values("timestamp")

    return df_out


# ==============================
# STATISTICHE
# ==============================

def compute_stats(returns):
    if not returns:
        return {
            "trades": 0,
            "winrate": 0.0,
            "pnl_total": 0.0,
            "pnl_mean": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "equity_final": 1.0,
        }

    rets = np.array(returns, dtype=float)
    wins = (rets > 0).sum()
    trades = len(rets)
    pnl_total = rets.sum()
    pnl_mean = rets.mean()

    eq = np.cumprod(1.0 + rets)
    equity_final = float(eq[-1])
    max_dd = 0.0
    peak = eq[0]
    for x in eq:
        if x > peak:
            peak = x
        dd = (x - peak) / peak
        if dd < max_dd:
            max_dd = dd

    std = rets.std(ddof=1)
    if std > 0:
        sharpe = pnl_mean / std * np.sqrt(trades)
    else:
        sharpe = 0.0

    winrate = wins / trades if trades > 0 else 0.0

    return {
        "trades": trades,
        "winrate": float(winrate),
        "pnl_total": float(pnl_total),
        "pnl_mean": float(pnl_mean),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_dd),
        "equity_final": float(equity_final),
    }


# ==============================
# BACKTEST PER SINGOLO SIMBOLO
# ==============================

def backtest_symbol(symbol: str):
    print("\n====================================================")
    print(f"   BACKTEST META-COORDINATOR: {symbol}")
    print("====================================================")

    # 1) CSV 1m
    csv_1m = os.path.join(DATA_1M_DIR, f"{symbol}_1m.csv")
    if not os.path.exists(csv_1m):
        print(f"[{symbol}] CSV 1m non trovato: {csv_1m}")
        return

    df_1m = pd.read_csv(csv_1m)
    df_1m["timestamp"] = pd.to_datetime(df_1m["timestamp"])
    df_1m = df_1m.sort_values("timestamp").reset_index(drop=True)

    # 2) Precompute prob daily / 1h / 15m
    df_daily_prob = compute_daily_probs(symbol)
    df_1h_prob = compute_1h_probs(symbol, df_1m)
    df_15m_prob = compute_15m_probs(symbol, df_1m)

    # 3) Base 1m con close
    base = df_1m[["timestamp", "close"]].copy().sort_values("timestamp")

    # asof merge: daily
    df_daily_prob = df_daily_prob.sort_values("timestamp")
    base = pd.merge_asof(
        base,
        df_daily_prob,
        on="timestamp",
        direction="backward",
    )

    # asof merge: 1h
    df_1h_prob = df_1h_prob.sort_values("timestamp")
    base = pd.merge_asof(
        base.sort_values("timestamp"),
        df_1h_prob,
        on="timestamp",
        direction="backward",
    )

    # asof merge: 15m
    df_15m_prob = df_15m_prob.sort_values("timestamp")
    base = pd.merge_asof(
        base.sort_values("timestamp"),
        df_15m_prob,
        on="timestamp",
        direction="backward",
    )

    base = base.dropna(subset=["p_ema1d", "p_sma1d", "p_1h", "p_15m"]).reset_index(drop=True)
    print(f"[{symbol}] Barre utilizzabili nel meta-backtest: {len(base)}")

    thr_ema, thr_sma = load_daily_thresholds(symbol)

    state = PositionState()
    equity = 1.0
    returns = []
    open_trades = 0
    closed_trades = 0

    for _, row in base.iterrows():
        price = float(row["close"])
        p_ema1d = float(row["p_ema1d"])
        p_sma1d = float(row["p_sma1d"])
        p_1h = float(row["p_1h"])
        p_15m = float(row["p_15m"])

        action, new_state = decide_action(
            symbol,
            price,
            p_ema1d,
            p_sma1d,
            thr_ema,
            thr_sma,
            p_1h,
            p_15m,
            state,
        )

        if action == "CLOSE" and state.direction != PositionDir.FLAT:
            if state.direction == PositionDir.LONG:
                ret = (price / state.entry_price - 1.0) - FEE
            else:
                ret = (state.entry_price / price - 1.0) - FEE
            returns.append(ret)
            equity *= (1.0 + ret)
            closed_trades += 1

        if action in ("OPEN_LONG", "OPEN_SHORT"):
            open_trades += 1

        state = new_state

    # chiusura forzata a fine periodo
    if state.direction != PositionDir.FLAT:
        price = float(base.iloc[-1]["close"])
        if state.direction == PositionDir.LONG:
            ret = (price / state.entry_price - 1.0) - FEE
        else:
            ret = (state.entry_price / price - 1.0) - FEE
        returns.append(ret)
        equity *= (1.0 + ret)
        closed_trades += 1

    stats = compute_stats(returns)

    print(f"[{symbol}] Trades aperti   : {open_trades}")
    print(f"[{symbol}] Trades chiusi   : {closed_trades}")
    print(f"[{symbol}] Trades totali   : {stats['trades']}")
    print(f"[{symbol}] Win-rate        : {stats['winrate']:.3f}")
    print(f"[{symbol}] PnL totale      : {stats['pnl_total']:.4f}")
    print(f"[{symbol}] PnL medio/trade : {stats['pnl_mean']:.5f}")
    print(f"[{symbol}] Sharpe/trade    : {stats['sharpe']:.3f}")
    print(f"[{symbol}] Max drawdown    : {stats['max_drawdown']:.3f}")
    print(f"[{symbol}] Equity finale   : {stats['equity_final']:.4f}")


# ==============================
# MAIN
# ==============================

def main():
    for sym in SYMBOLS:
        try:
            backtest_symbol(sym)
        except FileNotFoundError as e:
            print(f"[{sym}] ERRORE: {e}")
        except Exception as e:
            print(f"[{sym}] ERRORE GENERICO: {e}")

if __name__ == "__main__":
    main()
