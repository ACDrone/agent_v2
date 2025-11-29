#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
papertrade_meta.py

Paper trading del meta-coordinator con:
- Modelli daily EMA50_1D + SMA200_1D (regime macro)
- Modello direzionale 1H (bias operativo, orizzonte 60m su dati 1m)
- Modello direzionale 15M (timing ingresso/uscita, orizzonte 15m su dati 1m)

Simula un conto unico in euro con:
- Capitale iniziale: 1000 €
- Leva fissa (es. 3x)
- Allocazione per trade: frazione dell'equity
- Più posizioni aperte in parallelo su simboli diversi
- Log completo di tutte le operazioni in papertrades_meta.csv
"""

import os
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, List

import numpy as np
import pandas as pd
import joblib

# ==============================
# CONFIG CONTO
# ==============================

START_CAPITAL = 1000.0   # euro
FEE = 0.0005             # 0.05% per trade (entrata+uscita approssimata)
LEVERAGE = 3.0           # leva per posizione
ALLOC_FRACTION = 0.10    # 10% dell'equity come margin per nuovo trade

# ==============================
# CONFIG DATI / MODELLI
# ==============================

DATA_1M_DIR = "data_1m"
DATA_1D_DIR = "data_1d"

MODEL_15M_DIR = "modelli"
MODEL_1H_DIR = "modelli_1h"
MODEL_EMA1D_DIR = "modelli_ema50"
MODEL_SMA1D_DIR = "modelli_sma200"

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "AVAXUSDT", "XRPUSDT"]

# Soglie 1H HIGH-CONFIDENCE
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

# Soglie 15M HIGH-CONFIDENCE
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

from features import add_features_15m, add_features_1h  # tuo features.py
from features_long_1d import add_features_daily        # long EMA/SMA daily

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
    size: float = 0.0  # qui sarà il notional in quote

@dataclass
class Position:
    symbol: str
    direction: PositionDir
    entry_price: float
    entry_time: pd.Timestamp
    notional: float  # in valuta quote (USDT)

@dataclass
class Account:
    balance: float
    positions: Dict[str, Position]
    last_price: Dict[str, float]

    @property
    def equity(self) -> float:
        """Balance + PnL unrealizzato di tutte le posizioni."""
        eq = self.balance
        for pos in self.positions.values():
            price_now = self.last_price.get(pos.symbol, pos.entry_price)
            if pos.direction == PositionDir.LONG:
                ret = (price_now / pos.entry_price - 1.0)
            else:
                ret = (pos.entry_price / price_now - 1.0)
            eq += pos.notional * ret
        return eq

    @property
    def margin_used(self) -> float:
        """Margine impegnato (notional / leverage per ogni posizione)."""
        total = 0.0
        for pos in self.positions.values():
            total += pos.notional / LEVERAGE
        return total


# ==============================
# REGIME / BIAS / SIGNAL
# ==============================

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


def compute_regime(p_ema1d: float, p_sma1d: float,
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


def compute_size_notional(account: Account,
                          p_ema1d: float,
                          p_sma1d: float,
                          p_1h: float,
                          p_15m: float) -> float:
    """
    Calcola il notional da usare per il trade in base alla confidenza e all'equity.
    """
    equity = account.equity

    conf_regime = abs(p_ema1d - 0.5) + abs(p_sma1d - 0.5)
    conf_1h     = abs(p_1h - 0.5)
    conf_15m    = abs(p_15m - 0.5)

    w1, w2, w3 = 0.5, 0.3, 0.2
    conf_total = w1 * conf_regime + w2 * conf_1h + w3 * conf_15m

    # moltiplicatore 1–3 in base alla confidenza
    mult = 1.0 + 2.0 * conf_total
    mult = max(0.5, min(mult, 3.0))

    # margine allocato per trade
    margin_for_trade = equity * ALLOC_FRACTION * mult
    if margin_for_trade <= 0:
        return 0.0

    notional = margin_for_trade * LEVERAGE
    return notional


def decide_action(symbol: str,
                  price_now: float,
                  timestamp: pd.Timestamp,
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

    regime = compute_regime(p_ema1d, p_sma1d, thr_ema, thr_sma)
    bias_1h = compute_bias_1h(symbol, p_1h)
    sig_15m = compute_signal_15m(symbol, p_15m)

    new_state = PositionState(**vars(state))  # copia superficiale

    # Aggiorna durata posizione (in minuti, visto che le barre sono 1m)
    if new_state.direction != PositionDir.FLAT:
        new_state.bars_held += 1

    # Aperture solo su multipli di 15 minuti (allineati al modello 15m)
    can_open = (timestamp.minute % 15 == 0)

    # Minimo tempo di hold prima di chiudere (15 minuti)
    MIN_HOLD_BARS = 15

    # ===========================
    # NESSUNA POSIZIONE APERTA
    # ===========================
    if new_state.direction == PositionDir.FLAT:
        if can_open:
            # LONG -> regime BULL + bias 1H LONG + segnale 15m LONG
            if (regime == Regime.BULL) and (bias_1h == Bias1H.LONG) and (sig_15m == Signal15M.LONG):
                new_state.direction = PositionDir.LONG
                new_state.entry_price = price_now
                new_state.entry_time = timestamp
                new_state.bars_held = 0
                return "OPEN_LONG", new_state

            # SHORT -> regime BEAR + bias 1H SHORT + segnale 15m SHORT
            if (regime == Regime.BEAR) and (bias_1h == Bias1H.SHORT) and (sig_15m == Signal15M.SHORT):
                new_state.direction = PositionDir.SHORT
                new_state.entry_price = price_now
                new_state.entry_time = timestamp
                new_state.bars_held = 0
                return "OPEN_SHORT", new_state

        return "NOOP", new_state

    # ===========================
    # POSIZIONE APERTA
    # ===========================
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
    csv_path = os.path.join(DATA_1D_DIR, f"{symbol}_1d.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV 1d mancante: {csv_path}")

    df_raw = pd.read_csv(csv_path)
    df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"])
    df_raw = df_raw.sort_values("timestamp")

    df_feat = add_features_daily(df_raw)

    if "timestamp" not in df_feat.columns:
        raise ValueError("add_features_daily deve mantenere 'timestamp'.")

    ts = pd.to_datetime(df_feat["timestamp"])
    X = df_feat.drop(columns=["timestamp"])

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
    df_feat = add_features_1h(df_1m)
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

    return pd.DataFrame({
        "timestamp": ts.values,
        "p_1h": p_1h,
    }).sort_values("timestamp")


def compute_15m_probs(symbol: str, df_1m: pd.DataFrame) -> pd.DataFrame:
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

    return pd.DataFrame({
        "timestamp": ts.values,
        "p_15m": p_15m,
    }).sort_values("timestamp")


# ==============================
# STATISTICHE
# ==============================

def compute_stats(returns: List[float]) -> dict:
    if not returns:
        return {
            "trades": 0,
            "winrate": 0.0,
            "pnl_total": 0.0,
            "pnl_mean": 0.0,
        }

    rets = np.array(returns, dtype=float)
    wins = (rets > 0).sum()
    trades = len(rets)
    pnl_total = rets.sum()
    pnl_mean = rets.mean()
    winrate = wins / trades if trades > 0 else 0.0

    return {
        "trades": trades,
        "winrate": float(winrate),
        "pnl_total": float(pnl_total),
        "pnl_mean": float(pnl_mean),
    }


# ==============================
# PAPERTRADING MULTI-SIMBOLO
# ==============================

def build_meta_df_for_symbol(symbol: str) -> pd.DataFrame:
    """Costruisce il dataframe meta (1m) per un singolo simbolo."""
    csv_1m = os.path.join(DATA_1M_DIR, f"{symbol}_1m.csv")
    if not os.path.exists(csv_1m):
        raise FileNotFoundError(f"[{symbol}] CSV 1m non trovato: {csv_1m}")

    df_1m = pd.read_csv(csv_1m)
    df_1m["timestamp"] = pd.to_datetime(df_1m["timestamp"])
    df_1m = df_1m.sort_values("timestamp").reset_index(drop=True)

    df_daily_prob = compute_daily_probs(symbol)
    df_1h_prob = compute_1h_probs(symbol, df_1m)
    df_15m_prob = compute_15m_probs(symbol, df_1m)

    base = df_1m[["timestamp", "close"]].copy().sort_values("timestamp")

    df_daily_prob = df_daily_prob.sort_values("timestamp")
    base = pd.merge_asof(
        base,
        df_daily_prob,
        on="timestamp",
        direction="backward",
    )

    df_1h_prob = df_1h_prob.sort_values("timestamp")
    base = pd.merge_asof(
        base.sort_values("timestamp"),
        df_1h_prob,
        on="timestamp",
        direction="backward",
    )

    df_15m_prob = df_15m_prob.sort_values("timestamp")
    base = pd.merge_asof(
        base.sort_values("timestamp"),
        df_15m_prob,
        on="timestamp",
        direction="backward",
    )

    base = base.dropna(subset=["p_ema1d", "p_sma1d", "p_1h", "p_15m"]).reset_index(drop=True)
    base["symbol"] = symbol
    return base


def papertrade_meta():
    # Precalcolo dei dataframe meta per ogni simbolo
    meta_dfs = []
    thr_map = {}
    for sym in SYMBOLS:
        try:
            print(f"[{sym}] Preparo meta-df…")
            base = build_meta_df_for_symbol(sym)
            meta_dfs.append(base)
            thr_map[sym] = load_daily_thresholds(sym)
        except Exception as e:
            print(f"[{sym}] ERRORE PREPARAZIONE: {e}")

    if not meta_dfs:
        print("Nessun simbolo disponibile per il papertrading.")
        return

    # Unione di tutti i simboli su una timeline condivisa
    df_all = pd.concat(meta_dfs, ignore_index=True)
    df_all = df_all.sort_values(["timestamp", "symbol"]).reset_index(drop=True)

    print(f"Totale eventi (tutte le crypto): {len(df_all)}")

    # Stato iniziale
    account = Account(balance=START_CAPITAL, positions={}, last_price={})
    states: Dict[str, PositionState] = {sym: PositionState() for sym in SYMBOLS}
    trade_logs: List[dict] = []
    per_symbol_returns: Dict[str, List[float]] = {sym: [] for sym in SYMBOLS}

    # Loop principale
    for _, row in df_all.iterrows():
        sym = row["symbol"]
        ts = pd.to_datetime(row["timestamp"])
        price = float(row["close"])
        p_ema1d = float(row["p_ema1d"])
        p_sma1d = float(row["p_sma1d"])
        p_1h = float(row["p_1h"])
        p_15m = float(row["p_15m"])

        # aggiorna last price per mark-to-market
        account.last_price[sym] = price

        if sym not in states:
            states[sym] = PositionState()

        state = states[sym]
        thr_ema, thr_sma = thr_map[sym]

        action, new_state = decide_action(
            sym,
            price,
            ts,
            p_ema1d,
            p_sma1d,
            thr_ema,
            thr_sma,
            p_1h,
            p_15m,
            state,
        )

        # CHIUSURA POSIZIONE
        if action == "CLOSE" and state.direction != PositionDir.FLAT and sym in account.positions:
            pos = account.positions.pop(sym)

            if state.direction == PositionDir.LONG:
                ret = (price / pos.entry_price - 1.0) - FEE
            else:
                ret = (pos.entry_price / price - 1.0) - FEE

            pnl = pos.notional * ret
            account.balance += pnl
            per_symbol_returns[sym].append(ret)

            trade_logs.append({
                "symbol": sym,
                "direction": "LONG" if state.direction == PositionDir.LONG else "SHORT",
                "entry_time": pos.entry_time,
                "exit_time": ts,
                "entry_price": pos.entry_price,
                "exit_price": price,
                "notional": pos.notional,
                "ret": ret,
                "pnl": pnl,
                "equity_after": account.equity,
            })

        # APERTURA POSIZIONE
        if action in ("OPEN_LONG", "OPEN_SHORT"):
            # calcolo notional in base alla confidenza e all'equity
            notional = compute_size_notional(account, p_ema1d, p_sma1d, p_1h, p_15m)
            if notional > 0.0:
                # controllo margine: margin_used + margin_new <= equity
                margin_new = notional / LEVERAGE
                if account.margin_used + margin_new <= account.equity:
                    direction = new_state.direction
                    pos = Position(
                        symbol=sym,
                        direction=direction,
                        entry_price=price,
                        entry_time=ts,
                        notional=notional,
                    )
                    account.positions[sym] = pos
                    new_state.size = notional  # opzionale
                else:
                    # non apriamo per margine insufficiente
                    new_state.direction = PositionDir.FLAT
                    new_state.entry_price = 0.0
                    new_state.entry_time = None
                    new_state.bars_held = 0

        states[sym] = new_state

    # Chiudi tutte le posizioni rimaste aperte a fine storico
    if len(account.positions) > 0:
        last_ts = df_all["timestamp"].max()
        for sym, pos in list(account.positions.items()):
            price = account.last_price.get(sym, pos.entry_price)
            if pos.direction == PositionDir.LONG:
                ret = (price / pos.entry_price - 1.0) - FEE
            else:
                ret = (pos.entry_price / price - 1.0) - FEE
            pnl = pos.notional * ret
            account.balance += pnl
            per_symbol_returns[sym].append(ret)
            trade_logs.append({
                "symbol": sym,
                "direction": "LONG" if pos.direction == PositionDir.LONG else "SHORT",
                "entry_time": pos.entry_time,
                "exit_time": last_ts,
                "entry_price": pos.entry_price,
                "exit_price": price,
                "notional": pos.notional,
                "ret": ret,
                "pnl": pnl,
                "equity_after": account.equity,
            })
            del account.positions[sym]

    # ==========================
    # REPORT FINALE
    # ==========================
    print("\n===============================")
    print("   RISULTATI PAPERTRADING")
    print("===============================")
    print(f"Capitale iniziale : {START_CAPITAL:.2f}")
    print(f"Capitale finale   : {account.equity:.2f}")
    print(f"Pnl totale        : {account.equity - START_CAPITAL:.2f}")
    print(f"Rendimento totale : {(account.equity / START_CAPITAL - 1) * 100:.2f}%")

    tot_trades = sum(len(v) for v in per_symbol_returns.values())
    print(f"Trades totali     : {tot_trades}")

    for sym in SYMBOLS:
        stats = compute_stats(per_symbol_returns[sym])
        print(f"\n[{sym}]")
        print(f"  Trades   : {stats['trades']}")
        print(f"  Win-rate : {stats['winrate']:.3f}")
        print(f"  PnL tot  : {stats['pnl_total']:.4f}")
        print(f"  PnL medio: {stats['pnl_mean']:.5f}")

    # Salva log completo
    if trade_logs:
        df_log = pd.DataFrame(trade_logs)
        df_log.sort_values("entry_time", inplace=True)
        df_log.to_csv("papertrades_meta.csv", index=False)
        print("\nLog delle operazioni salvato in papertrades_meta.csv")
    else:
        print("\nNessun trade eseguito.")


# ==============================
# MAIN
# ==============================

if __name__ == "__main__":
    papertrade_meta()
