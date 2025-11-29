#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
meta_core.py — V2.0 (Versione Corretta per papertrade_live4)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict
import pandas as pd

# CONFIGURAZIONE
DEFAULT_MIN_HOLD_BARS = 15
DEFAULT_OPEN_EVERY_N_MINUTES = 1
DEFAULT_LEVERAGE = 3.0
DEFAULT_ALLOC_FRACTION = 0.10
DEFAULT_HARD_STOP_PCT = 0.02
ENTRY_THRESHOLD = 0.35
EXIT_THRESHOLD  = 0.10

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
    highest_pnl_pct: float = 0.0

@dataclass
class Position:
    symbol: str
    direction: PositionDir
    entry_price: float
    entry_time: pd.Timestamp
    notional: float
    leverage: float = DEFAULT_LEVERAGE

@dataclass
class Account:
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

def compute_composite_score(p_ema1d: float, p_sma1d: float,
                            p_1h: float, p_15m: float) -> float:
    avg_daily = (p_ema1d + p_sma1d) / 2
    score_daily = (avg_daily - 0.5) * 2.0
    score_1h = (p_1h - 0.5) * 2.0
    score_15m = (p_15m - 0.5) * 2.0
    
    W_DAILY, W_1H, W_15M = 0.30, 0.50, 0.20
    composite = (score_daily * W_DAILY) + (score_1h * W_1H) + (score_15m * W_15M)
    return max(-1.0, min(1.0, composite))

def decide_action(symbol: str,
                  price_now: float,
                  timestamp: pd.Timestamp,
                  p_ema1d: float,
                  p_sma1d: float,
                  p_1h: float,
                  p_15m: float,
                  state: PositionState,
                  min_hold_bars: int = DEFAULT_MIN_HOLD_BARS,
                  open_every_n_minutes: int = DEFAULT_OPEN_EVERY_N_MINUTES,
                  hard_stop_pct: float = DEFAULT_HARD_STOP_PCT
                  ) -> tuple[str, PositionState]:

    score = compute_composite_score(p_ema1d, p_sma1d, p_1h, p_15m)
    new_state = PositionState(
        direction=state.direction,
        entry_price=state.entry_price,
        entry_time=state.entry_time,
        bars_held=state.bars_held,
        size=state.size,
        highest_pnl_pct=state.highest_pnl_pct
    )

    if new_state.direction != PositionDir.FLAT:
        new_state.bars_held += 1

    can_open = (open_every_n_minutes <= 1 or (timestamp.minute % open_every_n_minutes == 0))

    # GESTIONE APERTA
    if new_state.direction != PositionDir.FLAT:
        current_pnl = 0.0
        if new_state.direction == PositionDir.LONG:
            current_pnl = (price_now - new_state.entry_price) / new_state.entry_price
        else:
            current_pnl = (new_state.entry_price - price_now) / new_state.entry_price
        
        if current_pnl > new_state.highest_pnl_pct:
            new_state.highest_pnl_pct = current_pnl

        if current_pnl < -hard_stop_pct:
            new_state.direction = PositionDir.FLAT
            return "CLOSE_STOP_LOSS", new_state

        if new_state.bars_held < min_hold_bars:
            return "HOLD", new_state

        if new_state.direction == PositionDir.LONG:
            if score < EXIT_THRESHOLD: 
                new_state.direction = PositionDir.FLAT
                return "CLOSE_WEAK_SCORE", new_state
            return "HOLD", new_state

        if new_state.direction == PositionDir.SHORT:
            if score > -EXIT_THRESHOLD:
                new_state.direction = PositionDir.FLAT
                return "CLOSE_WEAK_SCORE", new_state
            return "HOLD", new_state

    # GESTIONE FLAT
    if new_state.direction == PositionDir.FLAT:
        if not can_open:
            return "NOOP", new_state

        if score > ENTRY_THRESHOLD and p_1h > 0.55:
            new_state.direction = PositionDir.LONG
            new_state.entry_price = price_now
            new_state.entry_time = timestamp
            new_state.bars_held = 0
            new_state.highest_pnl_pct = 0.0
            return "OPEN_LONG", new_state

        if score < -ENTRY_THRESHOLD and p_1h < 0.45:
            new_state.direction = PositionDir.SHORT
            new_state.entry_price = price_now
            new_state.entry_time = timestamp
            new_state.bars_held = 0
            new_state.highest_pnl_pct = 0.0
            return "OPEN_SHORT", new_state

    return "NOOP", new_state

def compute_dynamic_leverage(p_ema1d: float, p_sma1d: float, p_1h: float, p_15m: float) -> int:
    score = compute_composite_score(p_ema1d, p_sma1d, p_1h, p_15m)
    abs_score = abs(score)
    if abs_score < 0.3:
        lev = 1 + int(abs_score * 10) 
    else:
        lev = 5 + int((abs_score - 0.3) * (35 / 0.7))
    return max(1, min(lev, 40))

def compute_size_notional(account: Account, p_ema1d: float, p_sma1d: float, p_1h: float, p_15m: float,
                          fixed_leverage: float = None, alloc_fraction: float = DEFAULT_ALLOC_FRACTION) -> float:
    equity = account.equity
    if equity <= 0: return 0.0
    leverage = fixed_leverage if fixed_leverage is not None else compute_dynamic_leverage(p_ema1d, p_sma1d, p_1h, p_15m)
    score = compute_composite_score(p_ema1d, p_sma1d, p_1h, p_15m)
    abs_score = abs(score)
    if abs_score < ENTRY_THRESHOLD: return 0.0
    size_mult = 0.5 + abs_score
    margin_for_trade = equity * alloc_fraction * size_mult
    if margin_for_trade <= 0: return 0.0
    return margin_for_trade * leverage