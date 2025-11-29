#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
papertrade_live.py

Paper trading LIVE del meta-coordinator.
- Logica identica all'originale.
- Aggiunta Leva Dinamica (importata da meta_core).
- Aggiunto Flush dei log per vedere i messaggi subito.
- Aggiunto calcolo Margine per la Dashboard.
"""

from __future__ import annotations

import os
import time
import json
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests
import joblib

# IMPORTIAMO LE FUNZIONI DAL TUO META_CORE
from meta_core import (
    Account, 
    Position, 
    PositionState, 
    PositionDir,
    compute_size_notional, 
    decide_action,
    compute_dynamic_leverage  # <--- Funzione che hai già nel tuo meta_core
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


# ======================================================
# UTILITY
# ======================================================

def log_msg(msg: str):
    """Stampa immediata a schermo (flush=True)"""
    print(msg, flush=True)

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
# CARICAMENTO MODELLI
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
    except: pass
    try:
        with open(os.path.join(MODEL_SMA1D_DIR, f"thr_sma200_1d_{symbol}.txt"), "r") as f:
            thr_sma = float(f.read().strip())
    except: pass
    return thr_ema, thr_sma


# ======================================================
# CALCOLI
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
    if len(df_1m) < 100: return None
    df_feat = add_features_1h(df_1m.copy())
    if df_feat.empty: return None
    feat_cols = [c for c in df_feat.columns if c not in ("timestamp", "target")]
    X = df_feat[feat_cols].apply(pd.to_numeric, errors="coerce")
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.dropna(inplace=True)
    if X.empty: return None
    X_last = X.iloc[[-1]]
    xgb_1h, lgb_1h = load_models_1h(symbol)
    p_x = xgb_1h.predict_proba(X_last)[:, 1][0]
    p_l = lgb_1h.predict_proba(X_last)[:, 1][0]
    return float(max(p_x, p_l))

def compute_last_p_15m(symbol: str, df_1m: pd.DataFrame) -> Optional[float]:
    if len(df_1m) < 100: return None
    df_feat = add_features_15m(df_1m.copy())
    if df_feat.empty: return None
    feat_cols = [c for c in df_feat.columns if c not in ("timestamp", "target")]
    X = df_feat[feat_cols].apply(pd.to_numeric, errors="coerce")
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.dropna(inplace=True)
    if X.empty: return None
    X_last = X.iloc[[-1]]
    xgb_15m, lgb_15m = load_models_15m(symbol)
    p_x = xgb_15m.predict_proba(X_last)[:, 1][0]
    p_l = lgb_15m.predict_proba(X_last)[:, 1][0]
    return float(max(p_x, p_l))


# ======================================================
# LOGGING
# ======================================================

def append_trade_log(row: dict, filename: str = TRADE_LOG_FILE):
    df = pd.DataFrame([row])
    if not os.path.exists(filename):
        df.to_csv(filename, index=False, mode="w", header=True)
    else:
        df.to_csv(filename, index=False, mode="a", header=False)

def append_history_log(rows: List[dict], filename: str = HISTORY_FILE):
    if not rows: return
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
# MAIN LOOP
# ======================================================

def sleep_to_next_minute():
    now = pd.Timestamp.utcnow()
    secs = 60 - now.second - now.microsecond / 1e6
    if secs < 1: secs = 1
    time.sleep(secs)

def bootstrap_1m_buffer(symbol: str) -> pd.DataFrame:
    log_msg(f"[{symbol}] Scarico storico 1m...")
    return fetch_binance_klines(symbol, interval="1m", limit=WARMUP_1M_BARS)

def papertrade_live():
    log_msg("=== AVVIO PAPER TRADE LIVE ===")
    
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
        except Exception as e:
            log_msg(f"[{sym}] ERRORE BOOTSTRAP: {e}")

    log_msg("=== LOOP LIVE AVVIATO ===")

    while True:
        loop_ts = pd.Timestamp.utcnow().floor("1min")
        log_msg(f"\n[LOOP] {loop_ts} UTC | Eq: {account.equity:.2f} | Posizioni: {len(account.positions)}")
        
        history_rows = []

        for sym in SYMBOLS:
            if sym not in buffers_1m: continue
            
            # 1. Update Dati
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

            last_row = df_buf.iloc[-1]
            ts_now = pd.to_datetime(last_row["timestamp"])
            price_now = float(last_row["close"])
            account.last_price[sym] = price_now

            # 2. Update Daily
            info = daily_info[sym]
            if ts_now.normalize() > info["last_day"]:
                try:
                    p_e, p_s, ts_d = compute_daily_probs_last(sym)
                    info.update({"p_ema1d": p_e, "p_sma1d": p_s, "last_day": ts_d.normalize()})
                    log_msg(f"[{sym}] Daily update: {p_e:.2f}, {p_s:.2f}")
                except Exception as e:
                    log_msg(f"[{sym}] Err daily: {e}")

            # 3. Predizioni
            try:
                p_1h = compute_last_p_1h(sym, df_buf)
                p_15m = compute_last_p_15m(sym, df_buf)
            except Exception as e:
                log_msg(f"[{sym}] Err pred: {e}")
                continue

            if p_1h is None or p_15m is None: continue
            last_p_1h[sym] = p_1h
            last_p_15m[sym] = p_15m
            
            history_rows.append({
                "timestamp": loop_ts, "symbol": sym, "price": price_now,
                "pred_15m": p_15m, "pred_1h": p_1h
            })

            # 4. Logica Trading
            state = states.get(sym, PositionState())
            action, new_state = decide_action(
                sym, price_now, ts_now,
                info["p_ema1d"], info["p_sma1d"],
                info["thr_ema"], info["thr_sma"],
                p_1h, p_15m, state
            )

            # 5. Esecuzione
            if action == "CLOSE":
                if sym in account.positions:
                    pos = account.positions.pop(sym)
                    ret = (price_now / pos.entry_price - 1.0) - FEE if state.direction == PositionDir.LONG else (pos.entry_price / price_now - 1.0) - FEE
                    pnl = pos.notional * ret
                    account.balance += pnl
                    trades_closed_count += 1
                    if pnl > 0: wins_count += 1
                    last_action_global = f"{sym} CLOSE"
                    
                    dur = int((ts_now - pos.entry_time).total_seconds()/60) if pos.entry_time else 0
                    trade_info = {
                        "symbol": sym, "side": "LONG" if state.direction == PositionDir.LONG else "SHORT",
                        "entry": pos.entry_price, "exit": price_now, "pnl": pnl, "minutes": dur
                    }
                    last_trades_cache.append(trade_info)
                    append_trade_log({**trade_info, "entry_time": pos.entry_time, "exit_time": ts_now, "notional": pos.notional, "equity_after": account.equity})
                    log_msg(f"[{sym}] CLOSE PnL={pnl:.2f}")

            if action in ("OPEN_LONG", "OPEN_SHORT"):
                
                # --- CHIAMATA LEVA DINAMICA (da meta_core) ---
                dyn_lev = compute_dynamic_leverage(info["p_ema1d"], info["p_sma1d"], p_1h, p_15m)
                
                notional = compute_size_notional(account, info["p_ema1d"], info["p_sma1d"], p_1h, p_15m, dyn_lev, ALLOC_FRACTION)
                
                if notional > 0 and (account.margin_used + notional/dyn_lev <= account.equity):
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
            if not price: continue
            pos = account.positions.get(sym)
            
            pnl_open, margin_used, lev = 0.0, 0.0, 1.0
            dir_str = "FLAT"
            
            if pos:
                lev = pos.leverage
                margin_used = pos.notional / lev
                if pos.direction == PositionDir.LONG:
                    pnl_open = (price / pos.entry_price - 1.0) * pos.notional
                    dir_str = "LONG"
                else:
                    pnl_open = (pos.entry_price / price - 1.0) * pos.notional
                    dir_str = "SHORT"
            
            market_snapshot[sym] = {
                "price": price, "p15": last_p_15m.get(sym, 0.5), "p1h": last_p_1h.get(sym, 0.5),
                "state": dir_str, "leverage": lev, "pnl": pnl_open, "margin": margin_used
            }

        save_state({
            "timestamp": str(loop_ts),
            "equity": account.equity, "margin_used": account.margin_used,
            "pnl_total": pnl_total, "drawdown": dd, "winrate": wr,
            "last_action": last_action_global, "markets": market_snapshot,
            "trades": last_trades_cache[-20:],
        })

        sleep_to_next_minute()

if __name__ == "__main__":
    papertrade_live()