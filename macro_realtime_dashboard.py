import pandas as pd
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import os
import re

# -----------------------------------------------------------
# CONFIGURAZIONE
# -----------------------------------------------------------
FOLDER_NAME = "data_1m"
FILE_PATTERN = "*.csv"

MACRO_SYMBOLS = {
    "QQQ": "Nasdaq",
    "ES=F": "SP500",
    "DX-Y.NYB": "DXY",
    "^VIX": "VIX",
    "^TNX": "US10Y"
}

TARGET_RESAMPLE = "1H"   # Resample coerente tra crypto e macro


# -----------------------------------------------------------
# PARSER ASSET ROBUSTO
# -----------------------------------------------------------
def parse_asset(filename):
    name = os.path.basename(filename).upper()

    # Cerca pattern tipo BTCUSDT, ETHUSD, SOLUSDT_1m ecc.
    m = re.match(r"([A-Z0-9]+?)(USDT|USD|PERP|_)", name)
    if m:
        return m.group(1)

    # Fallback: usa parte prima di "."
    return os.path.splitext(name)[0]


# -----------------------------------------------------------
# CARICAMENTO CRYPTO
# -----------------------------------------------------------
def load_from_folder():
    print(f"Scansione cartella '{FOLDER_NAME}'...")

    files = glob.glob(os.path.join(FOLDER_NAME, FILE_PATTERN))
    if not files:
        print("Nessun CSV trovato.")
        return pd.DataFrame()

    combined = []

    for fp in files:
        try:
            df = pd.read_csv(fp)
        except Exception as e:
            print(f"Errore {fp}: {e}")
            continue

        cols = {c.lower(): c for c in df.columns}

        # Data
        date_col = next((cols[c] for c in ['time', 'date', 'timestamp', 'open time', 'open_time'] if c in cols), None)
        close_col = next((cols[c] for c in ['close', 'close price', 'close_price'] if c in cols), None)

        if not date_col or not close_col:
            print(f"Saltato {fp}: colonne mancanti.")
            continue

        df["Date"] = pd.to_datetime(df[date_col], utc=True)
        df.set_index("Date", inplace=True)

        asset = parse_asset(fp)
        combined.append(df[close_col].rename(asset))

    if not combined:
        return pd.DataFrame()

    final_df = pd.concat(combined, axis=1).sort_index()
    return final_df


# -----------------------------------------------------------
# MACRO
# -----------------------------------------------------------
def fetch_macro_data(start_date, end_date):
    try:
        tickers = list(MACRO_SYMBOLS.keys())
        macro = yf.download(tickers, start=start_date, end=end_date, progress=False)["Close"]
        macro.rename(columns=MACRO_SYMBOLS, inplace=True)
        macro.index = pd.to_datetime(macro.index, utc=True)
        return macro
    except Exception as e:
        print("Errore Yahoo:", e)
        return pd.DataFrame()


# -----------------------------------------------------------
# ANALISI
# -----------------------------------------------------------
def analyze_correlation(crypto_df):
    if crypto_df.empty:
        print("Dati crypto vuoti.")
        return

    # Resample crypto per renderle confrontabili con macro
    crypto_res = crypto_df.resample(TARGET_RESAMPLE).last().ffill()

    print(f"Periodo crypto: {crypto_res.index.min()} → {crypto_res.index.max()}")

    # Scarica macro coerentemente
    macro_df = fetch_macro_data(
        crypto_res.index.min().date(),
        crypto_res.index.max().date()
    )

    if macro_df.empty:
        print("Macro vuote.")
        return

    # Resample macro e allinea
    macro_res = macro_df.resample(TARGET_RESAMPLE).last().ffill()

    # Join coerente
    full_df = crypto_res.join(macro_res, how="inner").dropna()

    if full_df.empty:
        print("Dati non sovrapponibili.")
        return

    # Rendimenti coerenti
    returns = full_df.pct_change().dropna()
    corr = returns.corr()

    # Heatmap
    plt.figure(figsize=(14, 10))
    sns.heatmap(
        corr, annot=True,
        cmap="RdBu", vmin=-1, vmax=1, center=0,
        fmt=".2f", linewidths=.5
    )
    plt.title("Correlazione Crypto vs Macro (Resample coerente)", fontsize=18)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------
# MAIN
# -----------------------------------------------------------
if __name__ == "__main__":
    crypto_data = load_from_folder()
    analyze_correlation(crypto_data)
