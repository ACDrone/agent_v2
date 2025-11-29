#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import requests
import pandas as pd

BASE_URL = "https://api.binance.com/api/v3/klines"

DATA_DIR = "data_1d"
os.makedirs(DATA_DIR, exist_ok=True)

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "AVAXUSDT", "XRPUSDT"]

# 5 anni di daily ~ 5*365 giorni
DAYS_1D = 5 * 365


def download_symbol_daily(symbol, days=DAYS_1D):
    print(f"\n=== Scarico DAILY {symbol} ({days} giorni, timeframe 1d) ===")

    end_ts = int(time.time() * 1000)
    start_ts = end_ts - days * 24 * 60 * 60 * 1000

    all_rows = []
    cur_start = start_ts

    while True:
        params = {
            "symbol": symbol,
            "interval": "1d",
            "startTime": cur_start,
            "limit": 1000
        }
        r = requests.get(BASE_URL, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()

        if not data:
            break

        all_rows.extend(data)

        last_open_time = data[-1][0]
        cur_start = last_open_time + 24 * 60 * 60 * 1000  # candela successiva

        if cur_start >= end_ts:
            break

        time.sleep(0.05)

    if not all_rows:
        print(f"Nessun dato per {symbol}")
        return

    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_volume", "taker_buy_quote_volume", "ignore"
    ]
    df = pd.DataFrame(all_rows, columns=cols)

    df["open"] = df["open"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)
    df["close"] = df["close"].astype(float)
    df["volume"] = df["volume"].astype(float)

    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms")

    df_out = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()

    out_path = os.path.join(DATA_DIR, f"{symbol}_1d.csv")
    df_out.to_csv(out_path, index=False)
    print(f"Salvato {out_path} con {len(df_out)} righe.")


if __name__ == "__main__":
    for sym in SYMBOLS:
        download_symbol_daily(sym)
