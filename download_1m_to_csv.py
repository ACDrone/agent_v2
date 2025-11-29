#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import requests
import pandas as pd

# =====================================================
# CONFIG
# =====================================================

# Cartella dove salvare i CSV a 1 minuto
DATA_DIR = "data_1m"
os.makedirs(DATA_DIR, exist_ok=True)

BASE_URL = "https://api.binance.com/api/v3/klines"

# Lista simboli che vuoi scaricare
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "AVAXUSDT", "XRPUSDT"]

# Numero di giorni di storico 1m da scaricare.
# Per EMA50_1h / SMA200_1h ha senso stare almeno su 180–365 giorni.
DAYS_1M = 365   # puoi portarlo a 180 se vuoi essere più leggero


def download_symbol(symbol, interval="1m", days=DAYS_1M):
    """
    Scarica 'days' giorni di candele da Binance per il simbolo e timeframe indicati
    e salva in data_1m/<symbol>_<interval>.csv

    Per i tuoi modelli:
      - 1m serve come base per TUTTO (15m, 1h, EMA/SMA lunghi via resample)
    """

    print(f"\n=== Scarico {symbol} ({days} giorni, timeframe {interval}) ===")

    end_ts = int(time.time() * 1000)                    # ora (ms)
    start_ts = end_ts - days * 24 * 60 * 60 * 1000      # days giorni fa (ms)

    all_rows = []
    cur_start = start_ts

    while True:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": cur_start,
            "limit": 1000  # max per chiamata
        }
        r = requests.get(BASE_URL, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()

        if not data:
            break

        all_rows.extend(data)

        last_open_time = data[-1][0]
        # passo alla candela successiva
        cur_start = last_open_time + 60_000 if interval == "1m" else last_open_time + 1_000

        # se abbiamo superato end_ts, fermati
        if cur_start >= end_ts:
            break

        # piccolo sleep per non martellare l'API
        time.sleep(0.1)

    if not all_rows:
        print(f"Nessun dato per {symbol}")
        return

    # costruiamo il DataFrame
    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_volume", "taker_buy_quote_volume", "ignore"
    ]
    df = pd.DataFrame(all_rows, columns=cols)

    # converto i tipi e preparo le colonne che servono
    df["open"] = df["open"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)
    df["close"] = df["close"].astype(float)
    df["volume"] = df["volume"].astype(float)

    # timestamp leggibile
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms")

    # tieni solo le colonne richieste dal modello
    df_out = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()

    out_path = os.path.join(DATA_DIR, f"{symbol}_{interval}.csv")
    df_out.to_csv(out_path, index=False)
    print(f"Salvato {out_path} con {len(df_out)} righe.")


if __name__ == "__main__":
    for sym in SYMBOLS:
        download_symbol(sym, interval="1m", days=DAYS_1M)
