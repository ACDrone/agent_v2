#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
papertrade_live.py

Paper trading LIVE del meta-coordinator:
- Gestisce esecuzione ordini e calcolo segnali.
- Salva stato istantaneo (JSON) per la dashboard.
- Salva storico prezzi/previsioni (CSV) per i grafici.
"""

from __future__ import annotations

import os
import time
import math
import json
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests
import joblib

from meta_core import (
    Account,
    Position,
    PositionState,
    PositionDir,
    compute_size_notional,
    decide_action,
)

from features import add_features_15m, add_features_1h
from features_long_1d import add_features_daily

# ======================================================
# CONFIG CONTO E SIMULAZIONE
# ======================================================

START_CAPITAL = 1000.0        # "euro"/USDT iniziali
FEE = 0.0005                  # 0.05% a trade (entrata+uscita approx)
LEVERAGE = 3.0                # leva per posizione
ALLOC_FRACTION = 0.10         # 10% dell'equity come margine base per trade

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "AVAXUSDT", "XRPUSDT"]

# directory modelli (coerenti con il tuo setup attuale)
MODEL_15M_DIR = "modelli"
MODEL_1H_DIR = "modelli_1h"
MODEL_EMA1D_DIR = "modelli_ema50"
MODEL_SMA1D_DIR = "modelli_sma200"

# per warm-up 1m
WARMUP_1M_BARS = 1500  # numero di barre iniziali 1m da scaricare

# FILE DI OUTPUT
TRADE_LOG_FILE = "papertrades_live_meta.csv"    # Log operazioni chiuse
HISTORY_FILE = "history_live.csv"               # Log prezzi e previsioni (per grafici dashboard)
STATE_FILE = "papertrade_state.json"            # Stato istantaneo dashboard

BINANCE_BASE = "https://api.binance.com"


# ======================================================
# UTILITY: FETCH BINANCE
# ======================================================

def fetch_binance_klines(symbol: str, interval: str = "1m", limit: int = 1000) -> pd.DataFrame:
    url = f"{BINANCE_BASE}/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit,
    }
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()

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
    df = fetch_binance_klines(symbol, interval="1d", limit=limit)
    return df


# ======================================================
# CARICAMENTO MODELLI
# ======================================================

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
        raise FileNotFoundError(f"Modelli EMA50_1d mancanti per {symbol}")
    xgb = joblib.load(xgb_path)
    lgb = joblib.load(lgb_path)
    return xgb, lgb


def load_models_sma1d(symbol: str):
    xgb_path = os.path.join(MODEL_SMA1D_DIR, f"xgb_sma200_1d_{symbol}.pkl")
    lgb_path = os.path.join(MODEL_SMA1D_DIR, f"lgb_sma200_1d_{symbol}.pkl")
    if not (os.path.exists(xgb_path) and os.path.exists(lgb_path)):
        raise FileNotFoundError(f"Modelli SMA200_1d mancanti per {symbol}")
    xgb = joblib.load(xgb_path)
    lgb = joblib.load(lgb_path)
    return xgb, lgb


def load_daily_thresholds(symbol: str) -> tuple[float, float]:
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


# ======================================================
# CALCOLO PROBABILITA' E FEATURES
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
# LOG E SALVATAGGIO
# ======================================================

def append_trade_log(row: dict, filename: str = TRADE_LOG_FILE):
    df = pd.DataFrame([row])
    if not os.path.exists(filename):
        df.to_csv(filename, index=False, mode="w", header=True)
    else:
        df.to_csv(filename, index=False, mode="a", header=False)


def append_history_log(rows: List[dict], filename: str = HISTORY_FILE):
    """
    Appende le righe di storico (prezzo e previsioni) al CSV.
    """
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
        print(f"[STATE] Errore salvataggio: {e}")


# ======================================================
# MAIN PAPERTRADING LIVE
# ======================================================

def sleep_to_next_minute():
    now = pd.Timestamp.utcnow()
    secs = 60 - now.second - now.microsecond / 1e6
    if secs < 1:
        secs = 1
    time.sleep(secs)


def bootstrap_1m_buffer(symbol: str) -> pd.DataFrame:
    print(f"[{symbol}] Scarico storico iniziale 1m ({WARMUP_1M_BARS} barre)…")
    df = fetch_binance_klines(symbol, interval="1m", limit=WARMUP_1M_BARS)
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def papertrade_live():
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
    print("=== BOOTSTRAP INIZIALE ===")
    for sym in SYMBOLS:
        try:
            df_1m = bootstrap_1m_buffer(sym)
            buffers_1m[sym] = df_1m
            p_ema1d, p_sma1d, ts_last_day = compute_daily_probs_last(sym)
            thr_ema, thr_sma = load_daily_thresholds(sym)
            daily_info[sym] = {
                "p_ema1d": p_ema1d,
                "p_sma1d": p_sma1d,
                "last_day": ts_last_day.normalize(),
                "thr_ema": thr_ema,
                "thr_sma": thr_sma,
            }
            if not df_1m.empty:
                account.last_price[sym] = float(df_1m.iloc[-1]["close"])
            print(f"[{sym}] Bootstrap OK. p_ema={p_ema1d:.3f}, p_sma={p_sma1d:.3f}")
        except Exception as e:
            print(f"[{sym}] ERRORE BOOTSTRAP: {e}")

    print("=== INIZIO LOOP LIVE (paper trading) ===")

    while True:
        loop_ts = pd.Timestamp.utcnow().floor("1min")
        print(f"\n[LOOP] {loop_ts} | Equity: {account.equity:.2f} | Posizioni: {len(account.positions)}")

        history_rows_to_save = [] # lista per il salvataggio CSV storico

        for sym in SYMBOLS:
            if sym not in buffers_1m:
                continue

            df_buf = buffers_1m[sym]
            last_ts = df_buf["timestamp"].max() if not df_buf.empty else None

            # 1) Fetch nuovi dati
            try:
                new_df = fetch_new_1m_since(sym, last_ts=last_ts, limit=1000)
            except Exception as e:
                print(f"[{sym}] ERRORE fetch_new_1m: {e}")
                continue

            if not new_df.empty:
                df_buf = pd.concat([df_buf, new_df], ignore_index=True)
                df_buf.sort_values("timestamp", inplace=True)
                if len(df_buf) > WARMUP_1M_BARS:
                    df_buf = df_buf.iloc[-WARMUP_1M_BARS:].reset_index(drop=True)
                buffers_1m[sym] = df_buf

            last_row = df_buf.iloc[-1]
            ts_now = pd.to_datetime(last_row["timestamp"])
            price_now = float(last_row["close"])
            account.last_price[sym] = price_now

            # 2) Aggiorna Daily
            info = daily_info.get(sym)
            if info is None: continue
            if ts_now.normalize() > info["last_day"]:
                try:
                    p_ema1d, p_sma1d, ts_last_day = compute_daily_probs_last(sym)
                    info.update({"p_ema1d": p_ema1d, "p_sma1d": p_sma1d, "last_day": ts_last_day.normalize()})
                    print(f"[{sym}] Aggiorno daily probs: {p_ema1d:.3f}, {p_sma1d:.3f}")
                except Exception as e:
                    print(f"[{sym}] ERRORE compute_daily: {e}")

            p_ema1d = info["p_ema1d"]
            p_sma1d = info["p_sma1d"]
            thr_ema = info["thr_ema"]
            thr_sma = info["thr_sma"]

            # 3) Features e Predizioni
            try:
                p_1h = compute_last_p_1h(sym, df_buf)
                p_15m = compute_last_p_15m(sym, df_buf)
            except Exception as e:
                print(f"[{sym}] ERRORE pred 1h/15m: {e}")
                continue

            if p_1h is None or p_15m is None:
                continue

            last_p_1h[sym] = p_1h
            last_p_15m[sym] = p_15m
            state = states.get(sym, PositionState())

            # Aggiungi riga allo storico per i grafici
            history_rows_to_save.append({
                "timestamp": loop_ts,
                "symbol": sym,
                "price": price_now,
                "pred_15m": p_15m,
                "pred_1h": p_1h
            })

            # 4) Logica Trading
            action, new_state = decide_action(
                symbol=sym,
                price_now=price_now,
                timestamp=ts_now,
                p_ema1d=p_ema1d,
                p_sma1d=p_sma1d,
                thr_ema=thr_ema,
                thr_sma=thr_sma,
                p_1h=p_1h,
                p_15m=p_15m,
                state=state,
            )

            # CLOSE
            if action == "CLOSE":
                if sym in account.positions:
                    pos = account.positions.pop(sym)
                    ret = (price_now / pos.entry_price - 1.0) - FEE if state.direction == PositionDir.LONG else (pos.entry_price / price_now - 1.0) - FEE
                    pnl = pos.notional * ret
                    account.balance += pnl
                    trades_closed_count += 1
                    if pnl > 0: wins_count += 1
                    last_action_global = f"{sym} CLOSE"
                    
                    dur_minutes = int((ts_now - pos.entry_time).total_seconds() / 60) if pos.entry_time else 0
                    last_trades_cache.append({
                        "symbol": sym, "side": "LONG" if state.direction == PositionDir.LONG else "SHORT",
                        "entry": float(pos.entry_price), "exit": float(price_now), "pnl": float(pnl), "minutes": dur_minutes
                    })
                    if len(last_trades_cache) > 50: last_trades_cache = last_trades_cache[-50:]
                    
                    append_trade_log({
                        "symbol": sym, "direction": "LONG" if state.direction == PositionDir.LONG else "SHORT",
                        "entry_time": pos.entry_time, "exit_time": ts_now,
                        "entry_price": pos.entry_price, "exit_price": price_now,
                        "notional": pos.notional, "ret": ret, "pnl": pnl, "equity_after": account.equity
                    })
                    print(f"[{sym}] CLOSE | PnL={pnl:.2f}")

            # OPEN
            if action in ("OPEN_LONG", "OPEN_SHORT"):
                notional = compute_size_notional(account, p_ema1d, p_sma1d, p_1h, p_15m, LEVERAGE, ALLOC_FRACTION)
                if notional > 0.0 and (account.margin_used + notional/LEVERAGE <= account.equity):
                    pos = Position(
                        symbol=sym, direction=new_state.direction,
                        entry_price=price_now, entry_time=ts_now,
                        notional=notional, leverage=LEVERAGE
                    )
                    account.positions[sym] = pos
                    new_state.size = notional
                    last_action_global = f"{sym} {action}"
                    print(f"[{sym}] {action} | Notional={notional:.2f}")
                else:
                    new_state.direction = PositionDir.FLAT
                    new_state.size = 0.0
                    print(f"[{sym}] {action} NO MARGIN")

            states[sym] = new_state

        # Fine ciclo simboli
        append_history_log(history_rows_to_save) # Salva CSV history

        equity_history.append(account.equity)
        max_eq = max(equity_history) if equity_history else account.equity
        pnl_total = account.equity - START_CAPITAL
        drawdown = (account.equity / max_eq - 1.0) if max_eq > 0 else 0.0
        winrate = wins_count / trades_closed_count if trades_closed_count > 0 else 0.0

        # Snapshot State
        market_snapshot = {}
        for sym in SYMBOLS:
            price = account.last_price.get(sym)
            if not price: continue
            
            st = states.get(sym, PositionState())
            pos = account.positions.get(sym)
            info = daily_info.get(sym, {})

            unreal_pnl = 0.0
            leverage_used = LEVERAGE # Default
            if pos:
                leverage_used = pos.leverage
                if st.direction == PositionDir.LONG:
                    unreal_pnl = (price / pos.entry_price - 1.0) * pos.notional
                else:
                    unreal_pnl = (pos.entry_price / price - 1.0) * pos.notional

            dir_str = "LONG" if st.direction == PositionDir.LONG else "SHORT" if st.direction == PositionDir.SHORT else "FLAT"

            market_snapshot[sym] = {
                "price": float(price),
                "p15": float(last_p_15m.get(sym, 0.5)),
                "p1h": float(last_p_1h.get(sym, 0.5)),
                "p_ema": float(info.get("p_ema1d", 0.5)),
                "p_sma": float(info.get("p_sma1d", 0.5)),
                "state": dir_str,
                "leverage": float(leverage_used),
                "minutes_hold": int(getattr(st, "bars_held", 0)),
                "pnl": float(unreal_pnl),
            }

        save_state({
            "timestamp": str(loop_ts),
            "equity": float(account.equity),
            "balance": float(account.balance),
            "margin_used": float(account.margin_used),
            "pnl_total": float(pnl_total),
            "drawdown": float(drawdown),
            "trades_closed": int(trades_closed_count),
            "winrate": float(winrate),
            "last_action": last_action_global,
            "markets": market_snapshot,
            "trades": last_trades_cache[-20:],
        })

        sleep_to_next_minute()

if __name__ == "__main__":
    papertrade_live()