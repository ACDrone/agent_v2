#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
meta_core.py — versione più aggressiva

- Daily (EMA50_1D + SMA200_1D) = regime/rischio
- 1H = bias operativo
- 15M = timing
- Apertura basata sulla maggioranza dei 3 livelli
- Size modulata dall’allineamento tra daily / 1H / 15M
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict

import pandas as pd


# ============================================================
# CONFIG DI SOGLIA (DEFAULT, PER-SYMBOL)
# ============================================================

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

# Parametri di gestione posizione
DEFAULT_MIN_HOLD_BARS = 15          # minimo 15 barre da 1m (~15 minuti)
DEFAULT_OPEN_EVERY_N_MINUTES = 1    # ora: puoi aprire ogni minuto

# Parametri di money management
DEFAULT_LEVERAGE = 3.0
DEFAULT_ALLOC_FRACTION = 0.10       # 10% dell'equity come margin base per trade


# ============================================================
# ENUM E STRUTTURE
# ============================================================

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
    """
    Stato logico di una posizione per un singolo simbolo
    (usato dal meta-coordinator per decidere OPEN/CLOSE/HOLD).
    """
    direction: PositionDir = PositionDir.FLAT
    entry_price: float = 0.0
    entry_time: Optional[pd.Timestamp] = None
    bars_held: int = 0
    size: float = 0.0  # notional associato (opzionale, usato in paper trading)


@dataclass
class Position:
    """
    Posizione effettiva nell'account (per paper trading/backtest).
    """
    symbol: str
    direction: PositionDir
    entry_price: float
    entry_time: pd.Timestamp
    notional: float  # esposizione in quote (es. USDT)
    leverage: float = DEFAULT_LEVERAGE


@dataclass
class Account:
    """
    Conto con balance (cash) + posizioni aperte.

    equity = balance + somma PnL non realizzato.
    margin_used = somma(notional / leverage) su tutte le posizioni.
    """
    balance: float
    positions: Dict[str, Position] = field(default_factory=dict)
    last_price: Dict[str, float] = field(default_factory=dict)

    @property
    def equity(self) -> float:
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
        total = 0.0
        for pos in self.positions.values():
            total += pos.notional / max(pos.leverage, 1e-9)
        return total


# ============================================================
# FUNZIONI DI INTERPRETAZIONE SEGNALI
# ============================================================

def compute_regime(p_ema1d: float, p_sma1d: float,
                   thr_ema: float, thr_sma: float) -> Regime:
    """
    Regime di lungo periodo basato su:
    - EMA50_1D (p_ema1d)
    - SMA200_1D (p_sma1d)

    Lo usiamo come "macro bias":
      - BULL se entrambi sopra soglia
      - BEAR se entrambi sotto (1 - soglia)
      - altrimenti NEUTRAL.
    """
    bull = (p_ema1d >= thr_ema) and (p_sma1d >= thr_sma)
    bear = (p_ema1d <= 1.0 - thr_ema) and (p_sma1d <= 1.0 - thr_sma)

    if bull and not bear:
        return Regime.BULL
    elif bear and not bull:
        return Regime.BEAR
    else:
        return Regime.NEUTRAL


def compute_bias_1h(symbol: str, p_1h: float) -> Bias1H:
    """Bias operativo 1H (long/short/neutral) in base alle soglie per simbolo."""
    hi = THR_HI_1H[symbol]
    lo = THR_LO_1H[symbol]

    if p_1h >= hi:
        return Bias1H.LONG
    elif p_1h <= lo:
        return Bias1H.SHORT
    else:
        return Bias1H.NEUTRAL


def compute_signal_15m(symbol: str, p_15m: float) -> Signal15M:
    """Segnale di timing 15M (long/short/none) in base alle soglie per simbolo."""
    hi = THR_HI_15M[symbol]
    lo = THR_LO_15M[symbol]

    if p_15m >= hi:
        return Signal15M.LONG
    elif p_15m <= lo:
        return Signal15M.SHORT
    else:
        return Signal15M.NONE


# ============================================================
# POSITION SIZING
# ============================================================

def compute_size_notional(account: Account,
                          p_ema1d: float,
                          p_sma1d: float,
                          p_1h: float,
                          p_15m: float,
                          leverage: float = DEFAULT_LEVERAGE,
                          alloc_fraction: float = DEFAULT_ALLOC_FRACTION) -> float:
    """
    Calcola il notional da usare per il trade.

    Daily modula il rischio:
    - daily allineato a 1H → size piena
    - daily neutrale       → size media
    - daily contro         → size ridotta
    """

    equity = account.equity
    if equity <= 0:
        return 0.0

    # direzione implicita LONG/SHORT rispetto a 0.5
    def sign_from_p(p: float) -> int:
        if p >= 0.55:
            return 1
        if p <= 0.45:
            return -1
        return 0

    dir_ema = sign_from_p(p_ema1d)
    dir_sma = sign_from_p(p_sma1d)
    dir_daily = dir_ema + dir_sma
    if dir_daily > 0:
        dir_daily = 1
    elif dir_daily < 0:
        dir_daily = -1
    else:
        dir_daily = 0

    dir_1h = 1 if p_1h >= 0.5 else -1
    dir_15m = 1 if p_15m >= 0.5 else -1

    # allineamento daily ↔ 1H
    if dir_daily == 0:
        align_regime = 0.7   # daily neutrale
    elif dir_daily == dir_1h:
        align_regime = 1.0   # daily allineato
    else:
        align_regime = 0.4   # daily contro

    # allineamento 1H ↔ 15M
    align_15m = 1.0 if dir_1h == dir_15m else 0.6

    # "confidenza" grezza: quanto i segnali si discostano da 0.5
    conf_regime = (abs(p_ema1d - 0.5) + abs(p_sma1d - 0.5)) * align_regime
    conf_1h     = abs(p_1h - 0.5)
    conf_15m    = abs(p_15m - 0.5) * align_15m

    # pesi: daily e 1H dominanti, 15M fine tuning
    w1, w2, w3 = 0.4, 0.4, 0.2
    conf_total = w1 * conf_regime + w2 * conf_1h + w3 * conf_15m

    # confidenza troppo bassa → non aprire
    if conf_total < 0.05:
        return 0.0

    # moltiplicatore 0.5–3 in base alla confidenza
    mult = 1.0 + 2.0 * conf_total
    mult = max(0.5, min(mult, 3.0))

    margin_for_trade = equity * alloc_fraction * mult
    if margin_for_trade <= 0:
        return 0.0

    notional = margin_for_trade * leverage
    return notional


# ============================================================
# FUNZIONE AUSILIARIA: VOTAZIONE DEI TRE LIVELLI
# ============================================================

def _votes(regime: Regime, bias_1h: Bias1H, sig_15m: Signal15M) -> int:
    v = 0
    if regime == Regime.BULL:
        v += 1
    elif regime == Regime.BEAR:
        v -= 1

    if bias_1h == Bias1H.LONG:
        v += 1
    elif bias_1h == Bias1H.SHORT:
        v -= 1

    if sig_15m == Signal15M.LONG:
        v += 1
    elif sig_15m == Signal15M.SHORT:
        v -= 1

    return v


# ============================================================
# DECISIONE DEL META-COORDINATOR
# ============================================================

def decide_action(symbol: str,
                  price_now: float,
                  timestamp: pd.Timestamp,
                  p_ema1d: float,
                  p_sma1d: float,
                  thr_ema: float,
                  thr_sma: float,
                  p_1h: float,
                  p_15m: float,
                  state: PositionState,
                  min_hold_bars: int = DEFAULT_MIN_HOLD_BARS,
                  open_every_n_minutes: int = DEFAULT_OPEN_EVERY_N_MINUTES
                  ) -> tuple[str, PositionState]:
    """
    Logica del meta-coordinator (versione più aggressiva).

    - Apertura: se la maggioranza (daily, 1H, 15M) è LONG/SHORT
      e almeno uno tra 1H e 15M è HIGH-CONFIDENCE in quella direzione.
    - Chiusura: dopo min_hold_bars, se la maggioranza non è più
      nella direzione del trade.
    """

    regime = compute_regime(p_ema1d, p_sma1d, thr_ema, thr_sma)
    bias_1h = compute_bias_1h(symbol, p_1h)
    sig_15m = compute_signal_15m(symbol, p_15m)

    # copia dello stato
    new_state = PositionState(
        direction=state.direction,
        entry_price=state.entry_price,
        entry_time=state.entry_time,
        bars_held=state.bars_held,
        size=state.size,
    )

    # aggiorna durata in barre 1m se in posizione
    if new_state.direction != PositionDir.FLAT:
        new_state.bars_held += 1

    # apertura solo se minuto % N == 0 (N=1 → sempre)
    can_open = (open_every_n_minutes <= 1 or
                (timestamp.minute % open_every_n_minutes == 0))

    # votazione dei tre livelli
    total_votes = _votes(regime, bias_1h, sig_15m)
    if total_votes > 0:
        majority_dir = 1   # bias complessivo LONG
    elif total_votes < 0:
        majority_dir = -1  # bias complessivo SHORT
    else:
        majority_dir = 0   # nessuna maggioranza

    # segnali short-term molto convinti?
    hi1h = THR_HI_1H[symbol]
    lo1h = THR_LO_1H[symbol]
    hi15 = THR_HI_15M[symbol]
    lo15 = THR_LO_15M[symbol]

    strong_long = (p_1h >= hi1h) or (p_15m >= hi15)
    strong_short = (p_1h <= lo1h) or (p_15m <= lo15)

    # =======================================================
    # NESSUNA POSIZIONE APERTA → valuta apertura
    # =======================================================
    if new_state.direction == PositionDir.FLAT:
        if not can_open:
            return "NOOP", new_state

        # OPEN LONG
        if majority_dir > 0 and strong_long:
            new_state.direction = PositionDir.LONG
            new_state.entry_price = price_now
            new_state.entry_time = timestamp
            new_state.bars_held = 0
            return "OPEN_LONG", new_state

        # OPEN SHORT
        if majority_dir < 0 and strong_short:
            new_state.direction = PositionDir.SHORT
            new_state.entry_price = price_now
            new_state.entry_time = timestamp
            new_state.bars_held = 0
            return "OPEN_SHORT", new_state

        return "NOOP", new_state

    # =======================================================
    # POSIZIONE APERTA → HOLD / CLOSE
    # =======================================================

    # vincolo minimo di tempo in posizione
    if new_state.bars_held < min_hold_bars:
        return "HOLD", new_state

    # ricalcolo majority_dir (per chiarezza)
    if total_votes > 0:
        majority_dir = 1
    elif total_votes < 0:
        majority_dir = -1
    else:
        majority_dir = 0

    # LONG: chiudi se la maggioranza non è più LONG
    if new_state.direction == PositionDir.LONG:
        if majority_dir <= 0:
            new_state.direction = PositionDir.FLAT
            return "CLOSE", new_state
        return "HOLD", new_state

    # SHORT: chiudi se la maggioranza non è più SHORT
    if new_state.direction == PositionDir.SHORT:
        if majority_dir >= 0:
            new_state.direction = PositionDir.FLAT
            return "CLOSE", new_state
        return "HOLD", new_state

    # fallback
    return "NOOP", new_state


def compute_dynamic_leverage(p_ema1d: float,
                             p_sma1d: float,
                             p_1h: float,
                             p_15m: float) -> int:
    """
    Determina la leva dinamica (1–40) in base alla confidenza dei modelli.
    Le leve alte compaiono solo in condizioni eccezionalmente forti.
    """

    # deviazioni da 0.5 = "confidenza"
    c_daily = abs(p_ema1d - 0.5) + abs(p_sma1d - 0.5)   # (0–1)
    c_1h    = abs(p_1h - 0.5)                           # (0–0.5)
    c_15m   = abs(p_15m - 0.5)                          # (0–0.5)

    # combinazione pesata
    score = (0.5 * c_daily) + (0.3 * c_1h) + (0.2 * c_15m)

    # score effettivo 0–1 → mappiamo a 1–40
    lev = int(1 + score * 39)

    # bound pulito
    lev = max(1, min(lev, 40))
    return lev
