#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BACKTEST SERIO 1H CON ZONA DI NO-TRADE

- Usa i modelli 1h (XGB + LGB) e la soglia thr_1h_<symbol>.txt
- Calcola p_final = max(p_xgb, p_lgb)
- Costruisce i ritorni LONG/SHORT a 60 minuti dalla barra di INGRESSO
- Motore sequenziale:
    * un solo trade alla volta
    * chiusura dopo HORIZON barre (60 x 1m)
    * zona di NO-TRADE:
        - se p_final >= thr_high  -> LONG
        - se p_final <= thr_low   -> SHORT
        - altrimenti              -> nessun trade
- Stampa:
    * configurazione BASE "all-in" (thr_high = thr_low = soglia del training)
    * configurazioni HIGH-CONFIDENCE (griglia di soglie hi/low)
"""

import os
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import joblib

from features_15m import add_features_pro

# ================================================================
# CONFIG
# ================================================================
DATA_DIR = "data_1m"
MODEL_DIR = "modelli_1h"

HORIZON_1H = 60      # 60 barre da 1m = 1 ora
FEE = 0.0005         # 0.05%

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "AVAXUSDT", "XRPUSDT"]

# k per simbolo per l'orizzonte 1h (gli stessi usati nel train_multi_1h.py)
K_MAP_1H = {
    "BTCUSDT": 1.0,
    "ETHUSDT": 1.0,
    "SOLUSDT": 1.2,
    "AVAXUSDT": 1.0,
    "XRPUSDT": 1.0,
}

# griglia per zona di confidenza (thr_high, thr_low)
CONF_LEVELS_1H: List[Tuple[float, float]] = [
    (0.55, 0.45),
    (0.60, 0.40),
    (0.65, 0.35),
    (0.70, 0.30),
    (0.75, 0.25),
]


# ================================================================
# UTILS: CARICA MODELLI + SOGLIA TRAINING
# ================================================================
def load_models_and_threshold_1h(symbol: str):
    xgb_path = os.path.join(MODEL_DIR, f"xgb_1h_{symbol}.pkl")
    lgb_path = os.path.join(MODEL_DIR, f"lgb_1h_{symbol}.pkl")
    thr_path = os.path.join(MODEL_DIR, f"thr_1h_{symbol}.txt")

    if not os.path.exists(xgb_path) or not os.path.exists(lgb_path):
        raise FileNotFoundError(f"[{symbol}][1h] Modelli non trovati in {MODEL_DIR}/")

    xgb = joblib.load(xgb_path)
    lgb = joblib.load(lgb_path)

    if os.path.exists(thr_path):
        with open(thr_path, "r") as f:
            thr = float(f.read().strip())
    else:
        thr = 0.5

    return xgb, lgb, thr


# ================================================================
# COSTRUISCI FEATURES IDENTICHE AL TRAINING (1H)
# ================================================================
def build_features_1h(symbol: str):
    csv_path = os.path.join(DATA_DIR, f"{symbol}_1m.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)

    df_raw = pd.read_csv(csv_path)
    df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"])
    df_raw = df_raw.sort_values("timestamp")

    # stesso k usato nel training 1h
    k = K_MAP_1H.get(symbol, 1.0)

    df_feat = add_features_pro(df_raw, horizon=HORIZON_1H, k=k)
    df_feat["timestamp"] = pd.to_datetime(df_feat["timestamp"])
    df_feat = df_feat.sort_values("timestamp")

    # filtro orario come nel training (12–21 UTC)
    df_feat["hour"] = df_feat["timestamp"].dt.hour
    df_feat = df_feat[df_feat["hour"].between(12, 21)]
    df_feat = df_feat.drop(columns=["hour"], errors="ignore")

    return df_raw, df_feat


# ================================================================
# PREDIZIONI ENSEMBLE
# ================================================================
def generate_predictions_1h(df_feat: pd.DataFrame, xgb, lgb) -> pd.DataFrame:
    df = df_feat.copy()

    feats = [c for c in df.columns if c not in ["timestamp", "target"]]
    X = df[feats]

    p_xgb = xgb.predict_proba(X)[:, 1]
    p_lgb = lgb.predict_proba(X)[:, 1]

    df["p_xgb"] = p_xgb
    df["p_lgb"] = p_lgb
    df["p_final"] = np.maximum(p_xgb, p_lgb)

    return df[["timestamp", "p_xgb", "p_lgb", "p_final"]]


# ================================================================
# COSTRUISCI DF BACKTEST (JOIN PREZZI + RITORNI)
# ================================================================
def build_backtest_df_1h(df_raw: pd.DataFrame, df_pred: pd.DataFrame) -> pd.DataFrame:
    # prezzi
    df_prices = df_raw[["timestamp", "close"]].copy()
    df_prices = df_prices.sort_values("timestamp")

    # predizioni
    df_pred = df_pred.sort_values("timestamp")

    # join su timestamp
    df = pd.merge(df_pred, df_prices, on="timestamp", how="inner")

    # prezzo futuro a HORIZON_1H barre
    df["future_price"] = df["close"].shift(-HORIZON_1H)

    # ritorni dalla BARRA DI INGRESSO (non dall'uscita)
    df["ret_long_entry"] = (df["future_price"] / df["close"] - 1.0) - FEE
    df["ret_short_entry"] = (df["close"] / df["future_price"] - 1.0) - FEE

    # pulizia
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # reindicizza
    return df.reset_index(drop=True)


# ================================================================
# MOTORE SEQUENZIALE 1H CON ZONA DI NO-TRADE
# ================================================================
def backtest_sequential_1h(df_bt: pd.DataFrame, thr_high: float, thr_low: float) -> Dict:
    """
    - Se non in trade:
        * se p_final >= thr_high  -> apre LONG
        * se p_final <= thr_low   -> apre SHORT
        * altrimenti              -> no-trade
    - Se in trade:
        * chiude dopo HORIZON_1H barre (ritorno calcolato dalla barra di ingresso)
    """
    in_trade = False
    entry_idx = None
    direction = None

    trades: List[Dict] = []
    pnl_total = 0.0
    wins = 0
    wins_long = 0
    wins_short = 0
    n_long = 0
    n_short = 0

    equity = 1.0
    equity_curve = [equity]

    for i in range(len(df_bt)):
        row = df_bt.iloc[i]

        if not in_trade:
            p = row["p_final"]

            if p >= thr_high:
                direction = "LONG"
                entry_idx = i
                in_trade = True
            elif p <= thr_low:
                direction = "SHORT"
                entry_idx = i
                in_trade = True

        else:
            # chiudiamo dopo HORIZON_1H barre
            if i - entry_idx >= HORIZON_1H:
                entry_row = df_bt.iloc[entry_idx]

                if direction == "LONG":
                    ret = float(entry_row["ret_long_entry"])
                    n_long += 1
                    if ret > 0:
                        wins_long += 1
                else:
                    ret = float(entry_row["ret_short_entry"])
                    n_short += 1
                    if ret > 0:
                        wins_short += 1

                pnl_total += ret
                if ret > 0:
                    wins += 1

                equity *= (1.0 + ret)
                equity_curve.append(equity)

                trades.append({
                    "entry_ts": entry_row["timestamp"],
                    "direction": direction,
                    "ret": ret,
                    "p_final": entry_row["p_final"],
                })

                in_trade = False
                entry_idx = None
                direction = None

    n_trades = len(trades)
    if n_trades > 0:
        winrate = wins / n_trades
        pnl_mean = pnl_total / n_trades
    else:
        winrate = 0.0
        pnl_mean = 0.0

    # win-rate per lato
    winrate_long = wins_long / n_long if n_long > 0 else 0.0
    winrate_short = wins_short / n_short if n_short > 0 else 0.0

    # max drawdown sulla equity curve
    max_dd = 0.0
    peak = equity_curve[0]
    for eq in equity_curve:
        if eq > peak:
            peak = eq
        dd = (eq / peak) - 1.0
        if dd < max_dd:
            max_dd = dd

    # Sharpe per trade (grezzo)
    rets = [t["ret"] for t in trades]
    if len(rets) > 1:
        mu = np.mean(rets)
        sigma = np.std(rets)
        sharpe = (mu / sigma) * np.sqrt(len(rets)) if sigma > 0 else 0.0
    else:
        sharpe = 0.0

    return {
        "trades": n_trades,
        "wins": wins,
        "winrate": winrate,
        "pnl_total": pnl_total,
        "pnl_mean": pnl_mean,
        "n_long": n_long,
        "n_short": n_short,
        "winrate_long": winrate_long,
        "winrate_short": winrate_short,
        "max_dd": max_dd,
        "sharpe": sharpe,
        "equity_final": equity,
        "thr_high": thr_high,
        "thr_low": thr_low,
    }


# ================================================================
# BACKTEST PER SINGOLO SIMBOLO (1H)
# ================================================================
def backtest_symbol_1h(symbol: str):
    print("\n====================================================")
    print(f"   BACKTEST SERIO 1H — {symbol}")
    print("====================================================")

    xgb, lgb, thr_train = load_models_and_threshold_1h(symbol)
    df_raw, df_feat = build_features_1h(symbol)

    # predizioni ensemble
    df_pred = generate_predictions_1h(df_feat, xgb, lgb)

    # df backtest con prezzi e ritorni
    df_bt = build_backtest_df_1h(df_raw, df_pred)

    if len(df_bt) == 0:
        print(f"[{symbol}][1h] Nessun dato valido per il backtest.")
        return

    print(f"[{symbol}][1h] Barre disponibili per backtest: {len(df_bt)}")
    print(f"[{symbol}][1h] Soglia training (ensemble): {thr_train:.3f}")

    # ------------------------------------------------------------
    # 1) CONFIGURAZIONE BASE: ALL-IN (nessuna no-trade zone)
    #    p >= thr_train → LONG, p <= thr_train → SHORT
    # ------------------------------------------------------------
    res_base = backtest_sequential_1h(df_bt, thr_high=thr_train, thr_low=thr_train)

    print("\n[BASE - ALL-IN]")
    print(f"  Trades totali     : {res_base['trades']}")
    print(f"  Win-rate          : {res_base['winrate']:.3f}")
    print(f"  PnL totale        : {res_base['pnl_total']:.4f}")
    print(f"  PnL medio/trade   : {res_base['pnl_mean']:.5f}")
    print(f"  Long trades       : {res_base['n_long']} (winrate {res_base['winrate_long']:.3f})")
    print(f"  Short trades      : {res_base['n_short']} (winrate {res_base['winrate_short']:.3f})")
    print(f"  Max drawdown      : {res_base['max_dd']:.4f}")
    print(f"  Sharpe (per trade): {res_base['sharpe']:.3f}")
    print(f"  Equity finale     : {res_base['equity_final']:.4f}")

    # ------------------------------------------------------------
    # 2) CONFIGURAZIONI HIGH-CONFIDENCE (zona no-trade)
    # ------------------------------------------------------------
    conf_results: List[Dict] = []
    for hi, lo in CONF_LEVELS_1H:
        res = backtest_sequential_1h(df_bt, thr_high=hi, thr_low=lo)
        conf_results.append(res)

    if conf_results:
        # best PnL
        best_pnl = max(conf_results, key=lambda r: r["pnl_total"])
        # best winrate con almeno 30 trade (per non avere numeri ridicoli)
        best_wr_candidates = [r for r in conf_results if r["trades"] >= 30]
        if best_wr_candidates:
            best_wr = max(best_wr_candidates, key=lambda r: r["winrate"])
        else:
            best_wr = max(conf_results, key=lambda r: r["winrate"])

        print("\n[HIGH-CONFIDENCE - MIGLIOR PnL]")
        print(f"  Soglie (hi/low)   : {best_pnl['thr_high']:.2f} / {best_pnl['thr_low']:.2f}")
        print(f"  Trades totali     : {best_pnl['trades']}")
        print(f"  Win-rate          : {best_pnl['winrate']:.3f}")
        print(f"  PnL totale        : {best_pnl['pnl_total']:.4f}")
        print(f"  PnL medio/trade   : {best_pnl['pnl_mean']:.5f}")
        print(f"  Max drawdown      : {best_pnl['max_dd']:.4f}")
        print(f"  Sharpe (per trade): {best_pnl['sharpe']:.3f}")
        print(f"  Equity finale     : {best_pnl['equity_final']:.4f}")

        print("\n[HIGH-CONFIDENCE - MIGLIOR WIN-RATE]")
        print(f"  Soglie (hi/low)   : {best_wr['thr_high']:.2f} / {best_wr['thr_low']:.2f}")
        print(f"  Trades totali     : {best_wr['trades']}")
        print(f"  Win-rate          : {best_wr['winrate']:.3f}")
        print(f"  PnL totale        : {best_wr['pnl_total']:.4f}")
        print(f"  PnL medio/trade   : {best_wr['pnl_mean']:.5f}")
        print(f"  Max drawdown      : {best_wr['max_dd']:.4f}")
        print(f"  Sharpe (per trade): {best_wr['sharpe']:.3f}")
        print(f"  Equity finale     : {best_wr['equity_final']:.4f}")


# ================================================================
# MAIN
# ================================================================
def main():
    for sym in SYMBOLS:
        backtest_symbol_1h(sym)


if __name__ == "__main__":
    main()
