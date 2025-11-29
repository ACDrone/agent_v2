#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
funding_carry_daily.py

Strategia SEMPLICE di funding carry daily su Binance Futures.

Idea:
- Per ogni giorno sommiamo i 3 funding (ogni 8h) del perpetual.
- Calcoliamo il funding annualizzato stimato.
- Se l'annualizzato in valore assoluto supera una soglia:
    - se funding > 0: "short perp / long spot" -> incassiamo funding_day
    - se funding < 0: "long perp / short spot" -> incassiamo |funding_day|
- PnL giornaliero ≈ |funding_day| - fee  (fee = costi/slippage stimati)
- Possibilità di:
    - applicare leva (leverage)
    - operare solo con funding positivo (long_only_positive)

ATTENZIONE:
- Modello didattico: ignora basis, commissioni reali, rischio exchange, ecc.
"""

import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================================
# CONFIGURAZIONE
# ============================================================================

CONFIG = {
    # Simboli futures Binance
    "symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],

    # Minimo funding annualizzato (in decimali) per attivare la strategia.
    # 0.05 = 5% annualizzato
    "min_annual_funding": 0.05,

    # Commissione/giorno quando siamo in posizione (slippage, fee, basis risk).
    # Per ora 0.0 per vedere l'edge "grezzo" del funding.
    "daily_fee": 0.0,

    # Numero massimo di funding da scaricare (max Binance = 1000).
    "funding_limit": 1000,

    # Leva applicata sui giorni attivi nella simulazione
    "leverage": 5.0,   # prova 2–3–5, niente follie

    # Se True: opera solo quando il funding è positivo (short perp / long spot).
    # Se False: opera anche con funding negativo (long perp / short spot).
    "long_only_positive": True,
}


# ============================================================================
# HELPER API
# ============================================================================

def fetch_binance_funding(symbol: str, limit: int = 1000) -> pd.DataFrame:
    """
    Scarica storico funding rate da Binance Futures (perpetual).
    Restituisce un DataFrame con:
        - timestamp (datetime naive)
        - fundingRate (float)
    """
    base_url = "https://fapi.binance.com/fapi/v1/fundingRate"
    params = {"symbol": symbol, "limit": limit}

    r = requests.get(base_url, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()

    if not isinstance(data, list) or len(data) == 0:
        raise RuntimeError(f"Nessun dato funding per {symbol}")

    df = pd.DataFrame(data)
    df["fundingRate"] = df["fundingRate"].astype(float)
    df["timestamp"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True).dt.tz_convert(None)
    df = df[["timestamp", "fundingRate"]].sort_values("timestamp")
    return df


def build_daily_funding(df_funding: pd.DataFrame) -> pd.DataFrame:
    """
    Converte funding 8h -> funding giornaliero.

    Output indicizzato per 'date' con:
        - funding_day  (somma funding 8h nel giorno)
        - funding_ann  (annualizzato ~ funding_day * 365)
    """
    df = df_funding.copy()
    df["date"] = df["timestamp"].dt.floor("D")

    daily = (
        df.groupby("date")["fundingRate"]
        .sum()
        .to_frame("funding_day")
        .sort_index()
    )

    daily["funding_ann"] = daily["funding_day"] * 365.0
    return daily


# ============================================================================
# BACKTEST CARRY
# ============================================================================

def backtest_funding_carry(
    daily: pd.DataFrame,
    min_annual: float,
    fee_per_day: float,
    leverage: float = 1.0,
    long_only_positive: bool = False,
) -> pd.DataFrame:
    """
    Strategia:
        - attiva trade solo se |funding_ann| >= min_annual
        - se funding_ann > 0:
              short perp / long spot -> PnL ~ +funding_day
          se funding_ann < 0:
              long perp / short spot -> PnL ~ -funding_day
        - se long_only_positive=True: ignora i giorni con funding negativo
        - ogni giorno attivo paghiamo 'fee_per_day'
        - PnL moltiplicato per 'leverage'
    """
    df = daily.copy()

    # soglia in valore assoluto
    df["active"] = df["funding_ann"].abs() >= min_annual

    df["direction"] = 0.0
    # funding > 0: short perp / long spot
    df.loc[df["funding_ann"] > 0, "direction"] = 1.0

    if long_only_positive:
        # funding < 0: non operiamo proprio
        df.loc[df["funding_ann"] < 0, "active"] = False
    else:
        # funding < 0: long perp / short spot
        df.loc[df["funding_ann"] < 0, "direction"] = -1.0

    # PnL grezzo = direction * funding_day * leverage
    raw_pnl = df["direction"] * df["funding_day"] * leverage

    # fee solo quando siamo attivi
    df["pnl"] = np.where(df["active"], raw_pnl - fee_per_day, 0.0)

    df["equity"] = (1.0 + df["pnl"]).cumprod()
    return df


def stats_from_equity(df: pd.DataFrame) -> dict:
    """
    Statistiche semplici:
        - total_return_x
        - days
        - mean_pnl
        - std_pnl
        - sharpe (grezzo, giornaliero * sqrt(365))
    """
    pnl = df["pnl"].values
    ret = df["equity"].iloc[-1] if len(df) > 0 else 1.0
    days = len(df)

    mean_pnl = float(np.mean(pnl)) if days > 0 else 0.0
    std_pnl = float(np.std(pnl)) if days > 1 else 0.0
    sharpe = (mean_pnl / std_pnl * np.sqrt(365)) if std_pnl > 0 else 0.0

    return {
        "total_return_x": ret,
        "days": days,
        "mean_pnl": mean_pnl,
        "std_pnl": std_pnl,
        "sharpe": sharpe,
    }


# ============================================================================
# MAIN PER SINGOLO SYMBOL
# ============================================================================

def process_symbol(symbol: str,
                   min_annual: float,
                   fee_per_day: float) -> pd.DataFrame:
    print(f"\n======================")
    print(f"PROCESSING {symbol}")
    print(f"======================")

    # 1) Funding 8h
    df_f = fetch_binance_funding(symbol, limit=CONFIG["funding_limit"])
    print(f"{symbol}: {len(df_f)} record di funding 8h")

    # 2) Aggregazione giornaliera
    daily = build_daily_funding(df_f)
    print(f"{symbol}: {len(daily)} giorni di funding aggregato")

    # 3) Backtest carry con leva e flag long_only_positive
    res = backtest_funding_carry(
        daily,
        min_annual,
        fee_per_day,
        leverage=CONFIG["leverage"],
        long_only_positive=CONFIG["long_only_positive"],
    )

    # Giorni in cui effettivamente la strategia è dentro
    active_days = int(res["active"].sum())
    print(f"{symbol}: giorni con posizione attiva = {active_days} su {len(res)}")

    st = stats_from_equity(res)

    print(f"{symbol}: Return totale = {st['total_return_x']:.3f}x "
          f"su {st['days']} giorni")
    print(f"{symbol}: Sharpe grezzo = {st['sharpe']:.2f}")

    # 4) Plot equity
    plt.figure(figsize=(10, 4))
    plt.plot(res.index, res["equity"])
    plt.title(
        f"{symbol} – Funding Carry "
        f"(min_ann={min_annual*100:.1f}%, fee={fee_per_day*100:.3f}%, "
        f"lev={CONFIG['leverage']:.1f}x, "
        f\"long_only={'YES' if CONFIG['long_only_positive'] else 'NO'}\")"
    )
    plt.grid()
    plt.tight_layout()
    plt.show()

    return res


# ============================================================================
# MAIN MULTI-SYMBOL / PORTAFOGLIO
# ============================================================================

def main():
    all_equity = []

    min_annual = CONFIG["min_annual_funding"]
    fee = CONFIG["daily_fee"]

    print("CONFIGURAZIONE STRATEGIA")
    print(f"  Min annualized funding: {min_annual*100:.1f}%")
    print(f"  Fee/giorno quando attivo: {fee*100:.3f}%")
    print(f"  Leverage: {CONFIG['leverage']:.1f}x")
    print(f"  Long only positive funding: {CONFIG['long_only_positive']}")
    print(f"  Symbols: {', '.join(CONFIG['symbols'])}")

    for sym in CONFIG["symbols"]:
        try:
            res = process_symbol(sym, min_annual, fee)
            # mantengo solo pnl/equity per composizione portafoglio
            tmp = res[["pnl", "equity"]].copy()
            tmp.columns = [f"{sym}_pnl", f"{sym}_equity"]
            all_equity.append(tmp)
        except Exception as e:
            print(f"ERRORE su {sym}: {e}")

    if not all_equity:
        print("\nNessun simbolo processato con successo.")
        return

    # Allineamento serie su giorni comuni
    combined = pd.concat(all_equity, axis=1, join="inner").sort_index()

    pnl_cols = [c for c in combined.columns if c.endswith("_pnl")]
    combined["pnl_portfolio"] = combined[pnl_cols].mean(axis=1)
    combined["equity_portfolio"] = (1.0 + combined["pnl_portfolio"]).cumprod()

    st = stats_from_equity(
        combined[["pnl_portfolio", "equity_portfolio"]].rename(
            columns={"pnl_portfolio": "pnl", "equity_portfolio": "equity"}
        )
    )

    print("\n====== PORTAFOGLIO EQUIPONDERATO ======")
    print(f"Return totale = {st['total_return_x']:.3f}x "
          f"su {st['days']} giorni")
    print(f"Sharpe grezzo = {st['sharpe']:.2f}")

    plt.figure(figsize=(10, 4))
    plt.plot(combined.index, combined["equity_portfolio"])
    plt.title(
        "Portfolio – Funding Carry "
        f"(lev={CONFIG['leverage']:.1f}x, long_only={CONFIG['long_only_positive']})"
    )
    plt.grid()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
