import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# -----------------------------------------------------------
# CONFIGURAZIONE
# -----------------------------------------------------------
MY_CRYPTOS = ["BTC", "ETH", "SOL", "XRP", "AVAX"]

MACRO_TICKERS = {
    "QQQ": "NASDAQ",
    "^VIX": "VIX",
    "DX-Y.NYB": "DXY"
}

# Usiamo più dati per calcolare gli indicatori tecnici
TRAINING_PERIOD = "2y"

def get_data():
    print("⬇️ Scaricando dati e calcolando indicatori tecnici complessi...")
    tickers = [f"{c}-USD" for c in MY_CRYPTOS] + list(MACRO_TICKERS.keys())
    
    # Scarica
    df = yf.download(tickers, period=TRAINING_PERIOD, interval="1d", progress=False, auto_adjust=False)['Close']
    
    # Rinomina
    new_names = {}
    for t in tickers:
        if "-USD" in t:
            new_names[t] = t.split("-")[0]
        elif t in MACRO_TICKERS:
            new_names[t] = MACRO_TICKERS[t]
    df.rename(columns=new_names, inplace=True)
    return df

# -----------------------------------------------------------
# CALCOLATORI INDICATORI TECNICI (Manual Implementation)
# -----------------------------------------------------------
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def add_technical_features(df, target_crypto):
    data = df.copy()
    col = target_crypto
    
    # 1. Rendimenti Semplici
    returns = data.pct_change(fill_method=None)
    
    # --- FEATURES TECNICHE AVANZATE ---
    
    # A. RSI (14 giorni) - Ipercomprato/Ipervenduto
    # Se RSI > 70 spesso scende, se < 30 spesso sale
    data['F_RSI'] = compute_rsi(data[col], 14).shift(1)
    
    # B. Distanza dalla Media Mobile a 20 giorni (Trend Breve)
    sma20 = data[col].rolling(20).mean()
    # (Prezzo Ieri - Media) / Media -> Ci dice quanto siamo lontani dalla media in %
    data['F_Dist_SMA20'] = ((data[col].shift(1) - sma20.shift(1)) / sma20.shift(1))
    
    # C. Volatilità (Standard Deviation a 7 giorni)
    # Se la volatilità esplode, spesso c'è un'inversione o continuazione forte
    data['F_Volat'] = returns[col].rolling(7).std().shift(1)
    
    # D. Momentum (Rendimento ultimi 3 giorni)
    data['F_Mom3'] = returns[col].rolling(3).mean().shift(1)

    # --- FEATURES MACRO ---
    data['F_NASDAQ_Ret'] = returns['NASDAQ'].shift(1)
    data['F_VIX_Ret']    = returns['VIX'].shift(1)
    
    # --- TARGET ---
    # 1 = Sale, 0 = Scende
    data['Target'] = np.where(returns[col] > 0, 1, 0)
    
    data.dropna(inplace=True)
    return data

def run_boosted_ai():
    raw_data = get_data()
    
    print("\n" + "="*70)
    print(f"🚀 AI GRADIENT BOOSTING + ANALISI TECNICA")
    print("="*70)
    print(f"{'ASSET':<6} | {'SEGNALE':<10} | {'CONFIDENZA':<10} | {'ACCURATEZZA'} | {'RSI (Ieri)'}")
    print("-" * 70)
    
    for coin in MY_CRYPTOS:
        # Prepara dataset con indicatori
        df_coin = add_technical_features(raw_data, coin)
        
        feature_cols = [c for c in df_coin.columns if c.startswith('F_')]
        X = df_coin[feature_cols]
        y = df_coin['Target']
        
        # Split Temporale (non random)
        split = int(len(X) * 0.85) # Usiamo l'85% per training
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]
        
        # --- MODELLO AVANZATO: GRADIENT BOOSTING ---
        # Questo modello impara dagli errori precedenti. È più "intelligente" sui dati finanziari.
        model = GradientBoostingClassifier(
            n_estimators=150,      # Numero di alberi correttivi
            learning_rate=0.05,    # Impara lentamente per non "imparare a memoria" (overfitting)
            max_depth=3,           # Alberi non troppo profondi
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # Valutazione
        test_preds = model.predict(X_test)
        acc = accuracy_score(y_test, test_preds)
        
        # Previsione Domani
        last_row = df_coin.iloc[[-1]]
        last_features = last_row[feature_cols]
        pred = model.predict(last_features)[0]
        prob = model.predict_proba(last_features)[0][pred]
        
        # Recupera RSI per visualizzarlo
        last_rsi = last_features['F_RSI'].values[0]
        
        signal = "🟢 LONG" if pred == 1 else "🔴 SHORT"
        
        # Formattazione Output
        acc_str = f"{acc:.1%}"
        if acc > 0.55: acc_str += " ⭐" # Stella se il modello è buono
        
        print(f"{coin:<6} | {signal:<10} | {prob:.1%}     | {acc_str:<10}  | {last_rsi:.1f}")

    print("-" * 70)
    print("LEGENDA RSI: <30 (Ipervenduto/Rimbalzo probabile) | >70 (Ipercomprato/Drop probabile)")
    print("LEGENDA ACC: ⭐ = Modello con edge statistico (>55%).")

if __name__ == "__main__":
    run_boosted_ai()