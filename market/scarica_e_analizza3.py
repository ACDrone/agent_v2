import ccxt
import pandas as pd
import ta
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import precision_score

# ==========================================
# ⚙️ CONFIGURAZIONE "V9 PRO"
# ==========================================
SYMBOL = "ETH/USDT"     # Coppia Futures
TIMEFRAME = "1h"
LIMIT = 1500            # Binance Futures permette fino a 1500 candele

# RISK MANAGEMENT (TRAILING)
INITIAL_CAPITAL = 1000
FEE = 0.0005            # 0.05% (Fee Futures sono più basse dello Spot!)
SLIPPAGE = 0.0002
RISK_PER_TRADE = 0.03   # 3% Rischio
ATR_MULTIPLIER = 2.5    # Distanza Trailing Stop

# ==========================================
# 1. DATI FUTURES (OHLCV + FUNDING + OI)
# ==========================================
def fetch_futures_data(symbol, timeframe, limit):
    print(f"\n📥 Scarico dati FUTURES {timeframe} per {symbol}...")
    exchange = ccxt.binanceusdm() # <--- MODIFICA CHIAVE: USIAMO IL CLIENT FUTURES
    
    try:
        # A. Scarica Candele (Prezzo)
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        # B. Scarica Funding Rate (Storico è difficile, usiamo proxy attuale per semplicità script)
        # Nota: Scaricare lo storico preciso del Funding/OI richiede chiamate API separate e lente.
        # Per questo script didattico, simuleremo Funding/OI basandoci sul volume e prezzo
        # poiché l'API pubblica storica di Funding/OI è limitata senza chiave API.
        
        # PROXY OI (Open Interest Proxy): Volume * Volatilità
        # (Nella realtà useremmo exchange.fetch_open_interest_history)
        df['OI_Proxy'] = df['volume'] * (df['high'] - df['low'])
        
        print(f"✅ Dati scaricati: {len(df)} candele.")
        return df

    except Exception as e:
        print(f"❌ Errore scaricamento Futures: {e}")
        return pd.DataFrame()

# ==========================================
# 2. FEATURE ENGINEERING (CON FUNDING PROXY)
# ==========================================
def add_features_v9(df):
    df = df.copy()

    # --- A. INDICATORI TECNICI ---
    df['RSI'] = ta.momentum.rsi(df['close'], window=14)
    df['ATR'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
    df['ADX'] = ta.trend.adx(df['high'], df['low'], df['close'], window=14)
    
    # --- B. MICROSTRUCTURE PROXIES ---
    # Funding Rate Proxy (Se prezzo sale forte, funding tende a salire)
    df['Funding_Proxy'] = df['close'].pct_change().rolling(8).mean() * 100
    
    # Open Interest Change (Proxy)
    df['OI_Change'] = df['OI_Proxy'].pct_change()

    # --- C. TARGET ---
    # Target: Prezzo sale > Fee * 3
    df['Next_Ret'] = df['close'].shift(-1).pct_change()
    df['Target'] = (df['Next_Ret'] > (FEE * 3)).astype(int)

    # --- SHIFT (Input=Ieri, Target=Oggi) ---
    features = ['RSI', 'ATR', 'ADX', 'Funding_Proxy', 'OI_Change']
    for col in features:
        df[f"{col}_prev"] = df[col].shift(1)

    df.dropna(inplace=True)
    return df, [f"{c}_prev" for c in features]

# ==========================================
# 3. BACKTEST CON TRAILING STOP 🏃‍♂️
# ==========================================
def run_trailing_backtest(df, model, X_test):
    capital = INITIAL_CAPITAL
    equity = [capital]
    position = 0      # 0=Flat, 1=Long
    entry_price = 0
    stop_loss = 0     # Questo sarà mobile (Trailing)
    
    print("\n⚙️ Avvio Backtest con TRAILING STOP...")

    for i in range(len(X_test)):
        idx = X_test.index[i]
        row = df.loc[idx]
        
        # 1. GESTIONE POSIZIONE APERTA
        if position == 1:
            # Aggiorna Trailing Stop
            # Se il prezzo sale, alziamo lo Stop Loss. Non lo abbassiamo MAI.
            current_atr = X_test.iloc[i]['ATR_prev']
            new_sl_level = row['close'] - (current_atr * ATR_MULTIPLIER)
            
            if new_sl_level > stop_loss:
                stop_loss = new_sl_level # Trailing Up!
            
            # Check Exit (Il prezzo ha toccato lo Stop Loss dinamico?)
            if row['low'] <= stop_loss:
                # EXIT
                exit_price = stop_loss * (1 - SLIPPAGE)
                
                # Calcolo Profitto
                # Assumiamo size fissa (tutto il capitale) per semplicità calcolo curve
                shares = (capital / entry_price) 
                pnl = (exit_price - entry_price) * shares
                cost = (entry_price * shares * FEE) + (exit_price * shares * FEE)
                
                capital += (pnl - cost)
                position = 0 # Torniamo Flat
                # print(f"  🔻 Stop Hit a {stop_loss:.2f}")

        # 2. GESTIONE INGRESSO (Se Flat)
        if position == 0:
            feats = X_test.iloc[[i]]
            prob = model.predict_proba(feats)[0][1]
            adx = feats['ADX_prev'].values[0]
            
            # Entriamo se XGBoost è convinto (>55%) e c'è Trend (ADX > 25)
            if prob > 0.55 and adx > 25:
                # ENTRY
                entry_price = row['open'] * (1 + SLIPPAGE)
                atr = feats['ATR_prev'].values[0]
                
                # Setup Iniziale Stop Loss
                stop_loss = entry_price - (atr * ATR_MULTIPLIER)
                
                # Paghiamo fee ingresso
                shares = capital / entry_price
                capital -= (entry_price * shares * FEE)
                position = 1
                # print(f"  🟢 Long a {entry_price:.2f} (SL: {stop_loss:.2f})")

        equity.append(capital)

    return equity

# ==========================================
# 4. MAIN
# ==========================================
def main():
    # 1. Dati Futures
    df_raw = fetch_futures_data(SYMBOL, TIMEFRAME, LIMIT)
    if df_raw.empty: return

    # 2. Features
    df, features = add_features_v9(df_raw)
    
    # 3. Split
    split = int(len(df) * 0.80)
    X = df[features]
    y = df['Target']
    
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    
    # 4. XGBoost Training
    print(f"🧠 Addestramento XGBoost su {len(X_train)} ore...")
    model = XGBClassifier(n_estimators=600, learning_rate=0.005, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    # 5. Run Trailing Backtest
    eq = run_trailing_backtest(df, model, X_test)
    
    final = eq[-1]
    ret = ((final - INITIAL_CAPITAL)/INITIAL_CAPITAL)*100
    
    print("\n========================================")
    print(f"🚀 V9 PRO: FUTURES + TRAILING STOP")
    print("========================================")
    print(f"Capitale Iniziale: ${INITIAL_CAPITAL}")
    print(f"Capitale Finale:   ${final:.2f}")
    print(f"Ritorno Netto:     {ret:.2f}%")
    print("========================================")
    
    # Feature Importance
    print("\n🔍 Cosa guida il modello?")
    print(pd.Series(model.feature_importances_, index=features).sort_values(ascending=False).head(3))

if __name__ == "__main__":
    main()