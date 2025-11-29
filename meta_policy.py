# meta_policy.py

from enum import Enum
from dataclasses import dataclass

# ===================================
# ENUM E STATO
# ===================================

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
    entry_time: str = ""
    bars_held_15m: int = 0
    size: float = 0.0


# ===================================
# SOGLIE (da backtest / training)
# ===================================

# esempio: qui metti ciò che hai visto in backtest_multi_1h
THR_HI_1H = {
    "BTCUSDT": 0.75,
    "ETHUSDT": 0.75,
    "SOLUSDT": 0.70,
    "AVAXUSDT": 0.70,
    "XRPUSDT": 0.70,
}

THR_LO_1H = {
    "BTCUSDT": 0.25,
    "ETHUSDT": 0.25,
    "SOLUSDT": 0.30,
    "AVAXUSDT": 0.30,
    "XRPUSDT": 0.30,
}

# soglie 15m (più strette o uguali, da tarare)
THR_HI_15M = {
    "BTCUSDT": 0.70,
    "ETHUSDT": 0.70,
    "SOLUSDT": 0.72,
    "AVAXUSDT": 0.72,
    "XRPUSDT": 0.72,
}

THR_LO_15M = {
    "BTCUSDT": 0.30,
    "ETHUSDT": 0.30,
    "SOLUSDT": 0.28,
    "AVAXUSDT": 0.28,
    "XRPUSDT": 0.28,
}

# soglie EMA/SMA daily (puoi anche leggerle dai file thr_*.txt)
THR_EMA1D = {
    "BTCUSDT": 0.60,
    "ETHUSDT": 0.60,
    "SOLUSDT": 0.60,
    "AVAXUSDT": 0.60,
    "XRPUSDT": 0.60,
}

THR_SMA1D = {
    "BTCUSDT": 0.60,
    "ETHUSDT": 0.60,
    "SOLUSDT": 0.60,
    "AVAXUSDT": 0.60,
    "XRPUSDT": 0.60,
}


# ===================================
# FUNZIONI DI COMBINAZIONE
# ===================================

def compute_regime(symbol: str, p_ema1d: float, p_sma1d: float) -> Regime:
    thr_e = THR_EMA1D[symbol]
    thr_s = THR_SMA1D[symbol]

    bull = (p_ema1d >= thr_e) and (p_sma1d >= thr_s)
    bear = (p_ema1d <= 1.0 - thr_e) and (p_sma1d <= 1.0 - thr_s)

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


def compute_size(symbol: str, p_ema1d: float, p_sma1d: float, p_1h: float, p_15m: float,
                 base_size: float = 1.0, max_mult: float = 3.0) -> float:
    # conf da 0 a 0.5 circa per ciascun
    conf_regime = abs(p_ema1d - 0.5) + abs(p_sma1d - 0.5)   # max ~1
    conf_1h     = abs(p_1h - 0.5)                           # max 0.5
    conf_15m    = abs(p_15m - 0.5)                          # max 0.5

    # pesi (da tarare)
    w1, w2, w3 = 0.5, 0.3, 0.2

    conf_total = w1*conf_regime + w2*conf_1h + w3*conf_15m

    mult = 1.0 + 2.0 * conf_total   # tra ~1 e ~3 (limitiamo dopo)
    mult = max(0.5, min(mult, max_mult))

    return base_size * mult


# ===================================
# DECISIONE FINALE
# ===================================

def decide_action(symbol: str,
                  price_now: float,
                  p_ema1d: float,
                  p_sma1d: float,
                  p_1h: float,
                  p_15m: float,
                  state: PositionState) -> tuple[str, PositionState]:
    """
    Ritorna (azione, nuovo_state)
    azione ∈ {"OPEN_LONG", "OPEN_SHORT", "CLOSE", "HOLD", "NOOP"}
    """

    regime = compute_regime(symbol, p_ema1d, p_sma1d)
    bias_1h = compute_bias_1h(symbol, p_1h)
    sig_15m = compute_signal_15m(symbol, p_15m)

    new_state = PositionState(**vars(state))  # shallow copy

    # aggiornamento contatore barre se in posizione
    if new_state.direction != PositionDir.FLAT:
        new_state.bars_held_15m += 1

    # =====================================
    # NESSUNA POSIZIONE APERTA → ENTRATA
    # =====================================
    if new_state.direction == PositionDir.FLAT:
        # CASO LONG
        if (regime == Regime.BULL) and (bias_1h == Bias1H.LONG) and (sig_15m == Signal15M.LONG):
            size = compute_size(symbol, p_ema1d, p_sma1d, p_1h, p_15m)
            new_state.direction = PositionDir.LONG
            new_state.entry_price = price_now
            new_state.entry_time = "NOW"   # qui ci metti il timestamp
            new_state.bars_held_15m = 0
            new_state.size = size
            return "OPEN_LONG", new_state

        # CASO SHORT
        if (regime == Regime.BEAR) and (bias_1h == Bias1H.SHORT) and (sig_15m == Signal15M.SHORT):
            size = compute_size(symbol, p_ema1d, p_sma1d, p_1h, p_15m)
            new_state.direction = PositionDir.SHORT
            new_state.entry_price = price_now
            new_state.entry_time = "NOW"
            new_state.bars_held_15m = 0
            new_state.size = size
            return "OPEN_SHORT", new_state

        # altrimenti → no-trade
        return "NOOP", new_state

    # =====================================
    # POSIZIONE APERTA → GESTIONE / USCITA
    # =====================================

    MIN_HOLD_BARS = 2  # tieni almeno 2 barre da 15m (30 min), da tarare

    # LONG aperto
    if new_state.direction == PositionDir.LONG:
        # se troppo presto, HOLD
        if new_state.bars_held_15m < MIN_HOLD_BARS:
            return "HOLD", new_state

        # motivi di uscita:
        # 1) bias 1h non è più long
        if bias_1h != Bias1H.LONG:
            new_state.direction = PositionDir.FLAT
            return "CLOSE", new_state

        # 2) regime macro diventa BEAR
        if regime == Regime.BEAR:
            new_state.direction = PositionDir.FLAT
            return "CLOSE", new_state

        # 3) 15m dà segnale short forte
        if sig_15m == Signal15M.SHORT:
            new_state.direction = PositionDir.FLAT
            return "CLOSE", new_state

        # altrimenti HOLD
        return "HOLD", new_state

    # SHORT aperto
    if new_state.direction == PositionDir.SHORT:
        if new_state.bars_held_15m < MIN_HOLD_BARS:
            return "HOLD", new_state

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

    # fallback
    return "NOOP", new_state
