#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
papertrade_live.py

Paper trading LIVE del meta-coordinator:

- Usa meta_core.py per la logica di regime/bias/signal + decisione OPEN/CLOSE.
- Usa features.py (15m, 1h) e features_long_1d.py (daily) per le feature.
- Scarica dati 1m e 1d da Binance via REST (polling).
- Gestisce un conto unico (balance + posizioni aperte) in "USDT".
- Logga tutte le operazioni chiuse in papertrades_live_meta.csv.
- Salva lo stato corrente in papertrade_state.json per la dashboard.

ATTENZIONE:
- È una simulazione, NON manda ordini reali.
- Usa REST Binance pubblico, nessuna API key richiesta.
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

# file di log per il paper trading live
TRADE_LOG_FILE = "papertrades_live_meta.csv"

# file di stato per la dashboard
STATE_FILE = "papertrade_state.json"

BINANCE_BASE = "https://api.binance.com"


# ======================================================
# UTILITY: FETCH BINANCE
# ======================================================

def fetch_binance_klines(symbol: str, interval: str = "1m", limit: int = 1000) -> pd.DataFrame:
    """
    Scarica ultime `limit` candele Binance per symbol/interval.
    Ritorna un DataFrame con colonne:
      timestamp, open, high, low, close, volume
    (timestamp = open_time)
    """
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
        # kline: [open_time, open, high, low, close, volume, close_time, ...]
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


def fetch_new_1m_since(symbol: str,
                       last_ts: Optional[pd.Timestamp],
                       limit: int = 1000) -> pd.DataFrame:
    """
    Scarica le ultime candele 1m e filtra solo quelle con timestamp > last_ts.
    Se last_ts è None, ritorna tutto il blocco (fino a `limit`).
    """
    df = fetch_binance_klines(symbol, interval="1m", limit=limit)
    if last_ts is not None:
        df = df[df["timestamp"] > last_ts].copy()
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def fetch_daily_history(symbol: str, limit: int = 500) -> pd.DataFrame:
    """
    Scarica storico daily (1d) per symbol.
    """
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
    """
    Carica thr_ema e thr_sma dai file txt generati in training.
    Se mancano, usa default 0.6.
    """
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
# DAILY PROBS (EMA50_1d / SMA200_1d)
# ======================================================

def compute_daily_probs_last(symbol: str) -> tuple[float, float, pd.Timestamp]:
    """
    Calcola p_ema1d, p_sma1d per l'ULTIMA barra giornaliera disponibile.
    Ritorna (p_ema1d_last, p_sma1d_last, ts_last_day)
    """

    df_daily = fetch_daily_history(symbol, limit=500)
    if df_daily.empty:
        return 0.5, 0.5, pd.Timestamp.utcnow()

    df_feat = add_features_daily(df_daily.copy())
    if "timestamp" not in df_feat.columns:
        raise ValueError("add_features_daily deve mantenere 'timestamp'.")

    df_feat.sort_values("timestamp", inplace=True)
    df_feat.reset_index(drop=True, inplace=True)

    x = df_feat.drop(columns=[c for c in ["timestamp", "target"] if c in df_feat.columns])
    ts = pd.to_datetime(df_feat["timestamp"])

    xgb_ema, lgb_ema = load_models_ema1d(symbol)
    xgb_sma, lgb_sma = load_models_sma1d(symbol)

    p_ema = np.maximum(
        xgb_ema.predict_proba(x)[:, 1],
        lgb_ema.predict_proba(x)[:, 1],
    )
    p_sma = np.maximum(
        xgb_sma.predict_proba(x)[:, 1],
        lgb_sma.predict_proba(x)[:, 1],
    )

    p_ema_last = float(p_ema[-1])
    p_sma_last = float(p_sma[-1])
    ts_last = pd.to_datetime(ts.iloc[-1])

    return p_ema_last, p_sma_last, ts_last


# ======================================================
# FEATURE LIVE: PREVISIONE 1H / 15M PER L'ULTIMA BARRA
# ======================================================

def compute_last_p_1h(symbol: str,
                      df_1m: pd.DataFrame) -> Optional[float]:
    """
    Calcola p_1h (prob. LONG) per l'ULTIMA barra 1m dal buffer df_1m.
    Ritorna None se impossibile.
    """
    if len(df_1m) < 100:  # warm-up minimo
        return None

    # feature engineering
    df_feat = add_features_1h(df_1m.copy())
    if df_feat.empty or "timestamp" not in df_feat.columns:
        return None

    df_feat.sort_values("timestamp", inplace=True)
    df_feat.reset_index(drop=True, inplace=True)

    # prendi solo le colonne usate in training (no timestamp/target)
    feat_cols = [c for c in df_feat.columns if c not in ("timestamp", "target")]

    # forza tutto a numerico (float) per evitare dtype 'object'
    X = df_feat[feat_cols].apply(pd.to_numeric, errors="coerce")
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.dropna(inplace=True)
    if X.empty:
        return None

    X_last = X.iloc[[-1]]  # shape (1, n_features)

    xgb_1h, lgb_1h = load_models_1h(symbol)
    p_x = xgb_1h.predict_proba(X_last)[:, 1][0]
    p_l = lgb_1h.predict_proba(X_last)[:, 1][0]

    return float(max(p_x, p_l))



def compute_last_p_15m(symbol: str,
                       df_1m: pd.DataFrame) -> Optional[float]:
    """
    Calcola p_15m (prob. LONG) per l'ULTIMA barra 1m dal buffer df_1m.
    Ritorna None se impossibile.
    """
    if len(df_1m) < 100:  # warm-up minimo
        return None

    df_feat = add_features_15m(df_1m.copy())
    if df_feat.empty or "timestamp" not in df_feat.columns:
        return None

    df_feat.sort_values("timestamp", inplace=True)
    df_feat.reset_index(drop=True, inplace=True)

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
# LOG OPERAZIONI
# ======================================================

def append_trade_log(row: dict, filename: str = TRADE_LOG_FILE):
    """
    Appende una riga al CSV di log. Se il file non esiste, crea con header.
    """
    df = pd.DataFrame([row])
    if not os.path.exists(filename):
        df.to_csv(filename, index=False, mode="w", header=True)
    else:
        df.to_csv(filename, index=False, mode="a", header=False)


def save_state(state: dict, filename: str = STATE_FILE):
    """
    Salva lo stato corrente in JSON, per la dashboard (dashboard_live.py).
    """
    try:
        with open(filename, "w") as f:
            json.dump(state, f, default=str, indent=2)
    except Exception as e:
        print(f"[STATE] Errore salvataggio: {e}")


# ======================================================
# MAIN PAPERTRADING LIVE
# ======================================================

def sleep_to_next_minute():
    """
    Attende fino all'inizio del prossimo minuto (circa).
    """
    now = pd.Timestamp.utcnow()
    # secondi mancanti alla prossima "minute boundary"
    secs = 60 - now.second - now.microsecond / 1e6
    if secs < 1:
        secs = 1
    time.sleep(secs)


def bootstrap_1m_buffer(symbol: str) -> pd.DataFrame:
    """
    Scarica uno storico iniziale 1m per warm-up feature.
    """
    print(f"[{symbol}] Scarico storico iniziale 1m ({WARMUP_1M_BARS} barre)…")
    df = fetch_binance_klines(symbol, interval="1m", limit=WARMUP_1M_BARS)
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def papertrade_live():
    # Stato globale account
    account = Account(balance=START_CAPITAL)

    # Buffer 1m per ogni simbolo
    buffers_1m: Dict[str, pd.DataFrame] = {}

    # Stato per ogni simbolo
    states: Dict[str, PositionState] = {sym: PositionState() for sym in SYMBOLS}

    # Daily probs + soglie per ogni simbolo
    daily_info = {}  # sym -> dict(p_ema1d, p_sma1d, last_day, thr_ema, thr_sma)

    # Stats per dashboard
    equity_history: List[float] = []
    trades_closed_count = 0
    wins_count = 0
    last_action_global = ""
    last_trades_cache: List[dict] = []  # ultimi N trade chiusi
    last_p_1h: Dict[str, float] = {}
    last_p_15m: Dict[str, float] = {}

    # =========================================
    # BOOTSTRAP
    # =========================================
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

            # aggiorna last_price per mark-to-market
            if not df_1m.empty:
                account.last_price[sym] = float(df_1m.iloc[-1]["close"])

            print(f"[{sym}] Bootstrap OK. p_ema1d={p_ema1d:.3f}, p_sma1d={p_sma1d:.3f}")

        except Exception as e:
            print(f"[{sym}] ERRORE BOOTSTRAP: {e}")

    print("=== INIZIO LOOP LIVE (paper trading) ===")

    # =========================================
    # LOOP PRINCIPALE: ogni minuto
    # =========================================
    while True:
        loop_ts = pd.Timestamp.utcnow().floor("1min")
        print(f"\n[LOOP] {loop_ts} | Equity attuale: {account.equity:.2f} | Posizioni aperte: {len(account.positions)}")

        for sym in SYMBOLS:
            if sym not in buffers_1m:
                continue

            df_buf = buffers_1m[sym]
            last_ts = df_buf["timestamp"].max() if not df_buf.empty else None

            # 1) scarica nuove candele 1m
            try:
                new_df = fetch_new_1m_since(sym, last_ts=last_ts, limit=1000)
            except Exception as e:
                print(f"[{sym}] ERRORE fetch_new_1m: {e}")
                continue

            if new_df.empty:
                continue

            # aggiorna buffer (mantieni solo ultime WARMUP_1M_BARS bar per non esplodere)
            df_buf = pd.concat([df_buf, new_df], ignore_index=True)
            df_buf.sort_values("timestamp", inplace=True)
            if len(df_buf) > WARMUP_1M_BARS:
                df_buf = df_buf.iloc[-WARMUP_1M_BARS:].reset_index(drop=True)
            buffers_1m[sym] = df_buf

            # ultimo prezzo/timestamp
            last_row = df_buf.iloc[-1]
            ts_now = pd.to_datetime(last_row["timestamp"])
            price_now = float(last_row["close"])
            account.last_price[sym] = price_now

            # 2) aggiorna daily probs se è cambiato il giorno
            info = daily_info.get(sym)
            if info is None:
                continue

            if ts_now.normalize() > info["last_day"]:
                # nuovo giorno: ricalcola p_ema1d / p_sma1d
                try:
                    p_ema1d, p_sma1d, ts_last_day = compute_daily_probs_last(sym)
                    info["p_ema1d"] = p_ema1d
                    info["p_sma1d"] = p_sma1d
                    info["last_day"] = ts_last_day.normalize()
                    print(f"[{sym}] Aggiorno daily probs: p_ema1d={p_ema1d:.3f}, p_sma1d={p_sma1d:.3f}")
                except Exception as e:
                    print(f"[{sym}] ERRORE compute_daily_probs_last: {e}")

            p_ema1d = info["p_ema1d"]
            p_sma1d = info["p_sma1d"]
            thr_ema = info["thr_ema"]
            thr_sma = info["thr_sma"]

            # 3) calcola p_1h e p_15m per l'ultima barra
            try:
                p_1h = compute_last_p_1h(sym, df_buf)
                p_15m = compute_last_p_15m(sym, df_buf)
            except Exception as e:
                print(f"[{sym}] ERRORE feature/pred 1h/15m: {e}")
                continue

            if p_1h is None or p_15m is None:
                continue

            last_p_1h[sym] = p_1h
            last_p_15m[sym] = p_15m

            state = states.get(sym, PositionState())

            # 4) decidi azione meta
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

            # 5) GESTIONE CHIUSURA
            if action == "CLOSE":
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
                    last_action_global = f"{sym} CLOSE {'LONG' if state.direction == PositionDir.LONG else 'SHORT'}"

                    dur_minutes = 0
                    if pos.entry_time is not None:
                        dur_minutes = int((ts_now - pos.entry_time).total_seconds() / 60)

                    trade_summary = {
                        "symbol": sym,
                        "side": "LONG" if state.direction == PositionDir.LONG else "SHORT",
                        "entry": float(pos.entry_price),
                        "exit": float(price_now),
                        "pnl": float(pnl),
                        "minutes": dur_minutes,
                    }
                    last_trades_cache.append(trade_summary)
                    if len(last_trades_cache) > 50:
                        last_trades_cache = last_trades_cache[-50:]

                    log_row = {
                        "symbol": sym,
                        "direction": "LONG" if state.direction == PositionDir.LONG else "SHORT",
                        "entry_time": pos.entry_time,
                        "exit_time": ts_now,
                        "entry_price": pos.entry_price,
                        "exit_price": price_now,
                        "notional": pos.notional,
                        "ret": ret,
                        "pnl": pnl,
                        "equity_after": account.equity,
                    }
                    append_trade_log(log_row)

                    print(f"[{sym}] CLOSE {log_row['direction']} | ret={ret:.4f} | pnl={pnl:.2f} | equity={account.equity:.2f}")
                else:
                    print(f"[{sym}] CLOSE ma nessuna posizione in account (incoerenza).")

            # 6) GESTIONE APERTURA
            if action in ("OPEN_LONG", "OPEN_SHORT"):
                # calcolo notional in base alla confidenza e all'equity
                notional = compute_size_notional(
                    account,
                    p_ema1d=p_ema1d,
                    p_sma1d=p_sma1d,
                    p_1h=p_1h,
                    p_15m=p_15m,
                    leverage=LEVERAGE,
                    alloc_fraction=ALLOC_FRACTION,
                )

                if notional > 0.0:
                    margin_new = notional / LEVERAGE
                    if account.margin_used + margin_new <= account.equity:
                        direction = new_state.direction
                        pos = Position(
                            symbol=sym,
                            direction=direction,
                            entry_price=price_now,
                            entry_time=ts_now,
                            notional=notional,
                            leverage=LEVERAGE,
                        )
                        account.positions[sym] = pos
                        new_state.size = notional

                        last_action_global = f"{sym} {action}"

                        print(f"[{sym}] {action} | notional={notional:.2f} | price={price_now:.4f} | equity={account.equity:.2f}")
                    else:
                        # margine insufficiente: annulla apertura
                        new_state.direction = PositionDir.FLAT
                        new_state.entry_price = 0.0
                        new_state.entry_time = None
                        new_state.bars_held = 0
                        new_state.size = 0.0
                        print(f"[{sym}] {action} BLOCCATO per margine insufficiente (equity={account.equity:.2f})")

            states[sym] = new_state

        # fine ciclo su tutti i simboli
        # stampa riassunto sintetico
        print(f"[SUMMARY] Equity: {account.equity:.2f} | Balance: {account.balance:.2f} | Margin used: {account.margin_used:.2f}")

        # aggiorna stats per dashboard
        equity_history.append(account.equity)
        max_eq = max(equity_history) if equity_history else account.equity
        pnl_total = account.equity - START_CAPITAL
        drawdown = (account.equity / max_eq - 1.0) if max_eq > 0 else 0.0
        winrate = wins_count / trades_closed_count if trades_closed_count > 0 else 0.0

        # snapshot per simbolo
        market_snapshot = {}
        for sym in SYMBOLS:
            price = account.last_price.get(sym, None)
            info = daily_info.get(sym, {})
            st = states.get(sym, PositionState())
            pos = account.positions.get(sym)

            if price is None:
                continue

            dir_str = "FLAT"
            if st.direction == PositionDir.LONG:
                dir_str = "LONG"
            elif st.direction == PositionDir.SHORT:
                dir_str = "SHORT"

            minutes_hold = getattr(st, "bars_held", 0)  # approx: 1 bar = 1 min

            unreal_pnl = 0.0
            if pos is not None:
                if st.direction == PositionDir.LONG:
                    unreal_pnl = (price / pos.entry_price - 1.0) * pos.notional
                elif st.direction == PositionDir.SHORT:
                    unreal_pnl = (pos.entry_price / price - 1.0) * pos.notional

            market_snapshot[sym] = {
                "price": float(price),
                "p15": float(last_p_15m.get(sym, 0.5)),
                "p1h": float(last_p_1h.get(sym, 0.5)),
                "p_ema": float(info.get("p_ema1d", 0.5)),
                "p_sma": float(info.get("p_sma1d", 0.5)),
                "state": dir_str,
                "minutes_hold": int(minutes_hold),
                "pnl": float(unreal_pnl),
            }

        state_dict = {
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
            "trades": last_trades_cache[-20:],  # ultimi 20 per dashboard
        }

        save_state(state_dict)

        # aspetta prossimo minuto "vero"
        sleep_to_next_minute()


# ======================================================
# ENTRYPOINT
# ======================================================

if __name__ == "__main__":
    papertrade_live()
