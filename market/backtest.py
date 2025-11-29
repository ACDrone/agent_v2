#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Backtest migliorato:

- Modello non lineare: RandomForestClassifier
- Feature: Nasdaq + BTC + volumi + "proxy balene" (come prima)
- Training solo sui giorni con movimento forte di BTC (|future_ret| > MOVE_THR)
- Test su tutto il periodo out-of-sample (2023+)

Obiettivo: spremere più informazione da Nasdaq + volumi + balene.
"""

import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

# =========================
# CONFIG
# =========================
START_DATE   = "2017-01-01"
SPLIT_DATE   = "2023-01-01"
TICKER_BTC   = "BTC-USD"
TICKER_NDX   = "^NDX"
FEE_PER_SIDE = 0.0005   # 0.05% per cambio posizione

# Giorni usati per addestrare:
# consideriamo "movimento forte" se |future_ret| > 1%
MOVE_THR     = 0.01


# =========================
# DOWNLOAD DATI
# =========================
def download_data():
    print(">>> Scarico dati BTC e NDX da Yahoo Finance (daily)...")

    btc = yf.download(TICKER_BTC, start=START_DATE, progress=False, auto_adjust=False)
    ndx = yf.download(TICKER_NDX, start=START_DATE, progress=False, auto_adjust=False)

    if btc.empty or ndx.empty:
        raise RuntimeError("Errore: dati vuoti da Yahoo Finance.")

    # Close + Volume
    if isinstance(btc.columns, pd.MultiIndex):
        btc_close = btc['Close'].iloc[:, 0]
        btc_vol   = btc['Volume'].iloc[:, 0]
    else:
        btc_close = btc['Close']
        btc_vol   = btc['Volume']

    if isinstance(ndx.columns, pd.MultiIndex):
        ndx_close = ndx['Close'].iloc[:, 0]
        ndx_vol   = ndx['Volume'].iloc[:, 0]
    else:
        ndx_close = ndx['Close']
        ndx_vol   = ndx['Volume']

    df = pd.DataFrame({
        "btc_close": btc_close,
        "btc_volume": btc_vol,
        "ndx_close": ndx_close,
        "ndx_volume": ndx_vol
    }).dropna()

    return df


# =========================
# FEATURE ENGINEERING
# =========================
def build_dataset(df: pd.DataFrame):
    df = df.copy()

    # Ritorni %
    df["btc_ret"] = df["btc_close"].pct_change()
    df["ndx_ret"] = df["ndx_close"].pct_change()

    # Volatilità rolling
    df["btc_vol_5"] = df["btc_ret"].rolling(5).std()
    df["ndx_vol_5"] = df["ndx_ret"].rolling(5).std()

    # Momentum prezzi
    df["btc_mom_5"]  = df["btc_close"] / df["btc_close"].shift(5)  - 1
    df["btc_mom_10"] = df["btc_close"] / df["btc_close"].shift(10) - 1
    df["ndx_mom_5"]  = df["ndx_close"] / df["ndx_close"].shift(5)  - 1
    df["ndx_mom_10"] = df["ndx_close"] / df["ndx_close"].shift(10) - 1

    # ============
    # VOLUMI BTC
    # ============
    df["btc_vol_log"] = np.log(df["btc_volume"].replace(0, np.nan))

    vol_mean_20 = df["btc_vol_log"].rolling(20).mean()
    vol_std_20  = df["btc_vol_log"].rolling(20).std()
    df["btc_vol_z_20"] = (df["btc_vol_log"] - vol_mean_20) / vol_std_20

    df["btc_vol_change_1"] = df["btc_volume"].pct_change()

    # ============
    # PROXY BALENE
    # ============
    df["btc_abs_ret"] = df["btc_ret"].abs()
    absret_mean_20 = df["btc_abs_ret"].rolling(20).mean()
    absret_std_20  = df["btc_abs_ret"].rolling(20).std()
    df["btc_absret_z_20"] = (df["btc_abs_ret"] - absret_mean_20) / absret_std_20

    df["whale_vol_spike"]   = df["btc_vol_z_20"].clip(lower=0)
    df["whale_move_spike"]  = df["btc_absret_z_20"].clip(lower=0)

    # Target: direzione di BTC domani
    df["future_ret"] = df["btc_ret"].shift(-1)

    # Drop NaN (inizio serie + lookahead)
    df = df.dropna()

    feature_cols = [
        # NASDAQ
        "ndx_ret",
        "ndx_vol_5",
        "ndx_mom_5",
        "ndx_mom_10",
        # BTC prezzo
        "btc_ret",
        "btc_vol_5",
        "btc_mom_5",
        "btc_mom_10",
        # Volumi / balene
        "btc_vol_z_20",
        "btc_vol_change_1",
        "whale_vol_spike",
        "whale_move_spike",
    ]

    X = df[feature_cols].values
    y = (df["future_ret"] > 0).astype(int).values

    return df, X, y, feature_cols


# =========================
# TRAIN / TEST SPLIT
# =========================
def train_test_split_time(df, X, y, split_date=SPLIT_DATE):
    mask_train = df.index < split_date
    mask_test  = df.index >= split_date

    X_train = X[mask_train]
    y_train = y[mask_train]
    X_test  = X[mask_test]
    y_test  = y[mask_test]

    df_train = df.loc[mask_train].copy()
    df_test  = df.loc[mask_test].copy()

    print(f"\n> Train: {df_train.index[0].date()} → {df_train.index[-1].date()}  ({len(df_train)} giorni)")
    print(f"> Test : {df_test.index[0].date()} → {df_test.index[-1].date()}  ({len(df_test)} giorni)")

    return df_train, df_test, X_train, X_test, y_train, y_test


# =========================
# MODELLO
# =========================
def build_model():
    # RandomForest non lineare
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=400,
            max_depth=8,
            min_samples_leaf=20,
            n_jobs=-1,
            random_state=42
        ))
    ])
    return model


# =========================
# BACKTEST
# =========================
def backtest(df_test: pd.DataFrame, proba_up,
             threshold_long=0.55, threshold_short=0.45):

    df = df_test.copy()
    df["proba_up"] = proba_up

    signal = np.zeros_like(proba_up, dtype=float)
    signal[proba_up > threshold_long] = 1.0
    signal[proba_up < threshold_short] = -1.0
    df["signal"] = signal

    # Ritorno strategia
    df["strategy_ret"] = df["signal"] * df["future_ret"]

    # Costi transazione
    position_change = df["signal"].diff().fillna(0).abs()
    df["costs"] = position_change * FEE_PER_SIDE

    df["strategy_net"] = df["strategy_ret"] - df["costs"]

    # Buy & hold 1x
    df["bh_ret"] = df["future_ret"]

    df["cum_bh"]    = (1 + df["bh_ret"]).cumprod()
    df["cum_strat"] = (1 + df["strategy_net"]).cumprod()

    tot_bh    = (df["cum_bh"].iloc[-1] - 1) * 100
    tot_strat = (df["cum_strat"].iloc[-1] - 1) * 100

    print("\n=== RISULTATI BACKTEST (TEST SET) ===")
    print(f"Buy & Hold BTC : {tot_bh:.2f}%")
    print(f"Strategia ML   : {tot_strat:.2f}%")
    print(f"Numero giorni  : {len(df)}")
    print(f"Giorni con posizione != 0: {(df['signal'] != 0).sum()}")

    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df["cum_bh"], label="Buy & Hold BTC", alpha=0.6)
    plt.plot(df.index, df["cum_strat"], label="Strategia ML (RF, strong-move training)", linewidth=2)
    plt.title(f"BTC-USD – Strategia ML vs Buy & Hold (Test da {SPLIT_DATE})")
    plt.ylabel("Moltiplicatore Capitale (1.0 = pari)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return df


# =========================
# MAIN
# =========================
def main():
    # 1) Dati
    df_raw = download_data()

    # 2) Dataset
    df_full, X, y, feature_cols = build_dataset(df_raw)
    print(f"\nFeature usate ({len(feature_cols)}): {feature_cols}")
    print(f"Numero totale di esempi: {len(df_full)}")

    # 3) Train / Test
    df_train, df_test, X_train, X_test, y_train, y_test = train_test_split_time(df_full, X, y)

    # ---- TRAIN SOLO SU GIORNI CON MOVIMENTO FORTE ----
    strong_mask = (df_train["future_ret"].abs() > MOVE_THR)
    n_strong = strong_mask.sum()
    print(f"\nGiorni 'forte movimento' nel train (|future_ret| > {MOVE_THR:.3%}): {n_strong} / {len(df_train)}")

    if n_strong > 100:
        X_train_used = X_train[strong_mask.values]
        y_train_used = y_train[strong_mask.values]
        df_train_used = df_train[strong_mask].copy()
        print("→ Alleno SOLO su questi giorni forti.")
    else:
        # Se ce ne fossero troppo pochi, ripieghiamo su tutto il train
        X_train_used = X_train
        y_train_used = y_train
        df_train_used = df_train.copy()
        print("→ Troppi pochi giorni forti, alleno su tutto il train.")

    # 4) Modello
    model = build_model()
    print("\n>>> Addestramento modello RandomForest...")
    model.fit(X_train_used, y_train_used)

    # 5) Metriche classificazione (su tutto il test)
    proba_test = model.predict_proba(X_test)[:, 1]
    y_pred = (proba_test >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    try:
        auc = roc_auc_score(y_test, proba_test)
    except ValueError:
        auc = np.nan

    print("\n=== METRICHE CLASSIFICAZIONE (TEST) ===")
    print(f"Accuracy: {acc:.3f}")
    print(f"AUC-ROC : {auc:.3f}")
    print("Matrice di confusione (righe = vero, colonne = predetto [0,1]):")
    print(confusion_matrix(y_test, y_pred))

    # Analisi solo sui giorni di movimento forte nel TEST
    strong_test_mask = (df_test["future_ret"].abs() > MOVE_THR)
    if strong_test_mask.sum() > 0:
        acc_strong = accuracy_score(
            y_test[strong_test_mask.values],
            y_pred[strong_test_mask.values]
        )
        print("\n=== ANALISI GIORNI TEST CON MOVIMENTO FORTE ===")
        print(f"Giorni forti nel test: {strong_test_mask.sum()}")
        print(f"Accuracy su questi giorni: {acc_strong:.3f}")

    # 6) Backtest
    backtest(df_test, proba_test,
             threshold_long=0.55,
             threshold_short=0.45)


if __name__ == "__main__":
    main()
