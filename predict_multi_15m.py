#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import joblib
import numpy as np
import pandas as pd
from datetime import timedelta

from features_15m import add_features_pro

DATA_DIR = "data_1m"
MODEL_DIR = "modelli"
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)


# ============================================================
# Carica modelli + soglia
# ============================================================
def load_models_and_threshold(symbol):
    xgb = joblib.load(os.path.join(MODEL_DIR, f"xgb_15m_{symbol}.pkl"))
    lgb = joblib.load(os.path.join(MODEL_DIR, f"lgb_15m_{symbol}.pkl"))

    thr_path = os.path.join(MODEL_DIR, f"thr_15m_{symbol}.txt")
    if os.path.exists(thr_path):
        with open(thr_path, "r") as f:
            thr = float(f.read().strip())
    else:
        thr = 0.5

    return xgb, lgb, thr


# ============================================================
# Carica CSV raw + calcola feature identiche al training
# ============================================================
def load_and_process(symbol):
    path = os.path.join(DATA_DIR, f"{symbol}_1m.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    df_raw = pd.read_csv(path)
    df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"])
    df_raw = df_raw.sort_values("timestamp")

    df_feat = add_features_pro(df_raw)

    return df_raw, df_feat


# ============================================================
# Previsione singola (ultima candela)
# ============================================================
def make_prediction(symbol, xgb, lgb, thr, df_raw, df_feat):
    if len(df_feat) == 0:
        return None

    row = df_feat.iloc[-1]
    ts = row["timestamp"]

    # recupero prezzo
    mask = df_raw["timestamp"] == ts
    if mask.any():
        entry_price = df_raw.loc[mask, "close"].iloc[-1]
    else:
        entry_price = df_raw["close"].iloc[-1]

    # feature vector
    x = row.drop(labels=["target", "timestamp"], errors="ignore")
    x = pd.to_numeric(x, errors="coerce").to_frame().T

    p1 = xgb.predict_proba(x)[0, 1]
    p2 = lgb.predict_proba(x)[0, 1]
    p_final = max(p1, p2)

    # decisione direzionale
    if p_final > thr:
        signal = "LONG"
    else:
        signal = "SHORT"

    return {
        "timestamp": ts,
        "entry_price": float(entry_price),
        "p_xgb": float(p1),
        "p_lgb": float(p2),
        "p_final": float(p_final),
        "signal": signal
    }


# ============================================================
# Verifica risultato dopo HORIZON minuti
# ============================================================
def check_results(df_raw, preds, horizon=15):
    for p in preds:
        if "result" in p:
            continue

        ts_target = p["timestamp"] + timedelta(minutes=horizon)
        future_rows = df_raw[df_raw["timestamp"] >= ts_target]

        if len(future_rows) == 0:
            continue

        future_price = float(future_rows.iloc[0]["close"])
        p["future_price"] = future_price

        if p["signal"] == "LONG":
            p["result"] = "AZZECCATO" if future_price > p["entry_price"] else "SBAGLIATO"
        else:
            p["result"] = "AZZECCATO" if future_price < p["entry_price"] else "SBAGLIATO"


# ============================================================
# Scrivi log CSV
# ============================================================
def write_log(symbol, preds):
    if len(preds) == 0:
        return

    path = os.path.join(LOG_DIR, f"live_15m_{symbol}.csv")
    pd.DataFrame(preds).to_csv(path, index=False)


# ============================================================
# LOOP LIVE (o simulato)
# ============================================================
def main():
    if len(sys.argv) < 2:
        print("Uso: python predict_multi_15m.py BTCUSDT")
        sys.exit(1)

    symbol = sys.argv[1]
    xgb, lgb, thr = load_models_and_threshold(symbol)

    preds = []
    last_ts = None

    print(f"[{symbol}] Avvio live predictor — soglia {thr:.3f}")

    while True:
        try:
            df_raw, df_feat = load_and_process(symbol)

            current_ts = df_feat["timestamp"].iloc[-1]

            # verifica risultati trade precedenti
            check_results(df_raw, preds)

            # nessuna nuova candela
            if last_ts is not None and current_ts == last_ts:
                write_log(symbol, preds)
                time.sleep(60)
                continue

            # nuova candela → nuova previsione
            last_ts = current_ts
            pred = make_prediction(symbol, xgb, lgb, thr, df_raw, df_feat)

            if pred:
                preds.append(pred)
                print(
                    f"{pred['timestamp']} | {pred['signal']} | "
                    f"p={pred['p_final']:.3f} | price={pred['entry_price']:.2f}"
                )

            write_log(symbol, preds)
            time.sleep(60)

        except KeyboardInterrupt:
            print("\nTerminato.")
            break

        except Exception as e:
            print("Errore:", e)
            time.sleep(60)


if __name__ == "__main__":
    main()
