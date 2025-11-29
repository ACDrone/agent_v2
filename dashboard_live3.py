import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
import time

# ==========================================
# CONFIGURAZIONE
# ==========================================
st.set_page_config(
    page_title="🔥 CRYPTO COMMAND CENTER",
    layout="wide",
    page_icon="🦅",
    initial_sidebar_state="collapsed"
)

# LISTA ASSET (Deve coincidere con il bot)
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "AVAXUSDT", "XRPUSDT"]

# FILE
HISTORY_FILE = "history_live.csv"
TRADE_FILE = "papertrades_live_meta.csv"
STATE_FILE = "papertrade_state.json"

# Auto-refresh logic
if 'last_update' not in st.session_state:
    st.session_state.last_update = time.time()

def refresh():
    st.session_state.last_update = time.time()

# ==========================================
# LOADERS
# ==========================================
def load_state():
    if not os.path.exists(STATE_FILE): return {}
    try:
        with open(STATE_FILE, "r") as f: return json.load(f)
    except: return {}

def load_history():
    if not os.path.exists(HISTORY_FILE): return pd.DataFrame()
    try:
        df = pd.read_csv(HISTORY_FILE)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except: return pd.DataFrame()

def load_trades():
    if not os.path.exists(TRADE_FILE): return pd.DataFrame()
    try:
        df = pd.read_csv(TRADE_FILE)
        df['entry_time'] = pd.to_datetime(df['entry_time'])
        df['exit_time'] = pd.to_datetime(df['exit_time'])
        return df
    except: return pd.DataFrame()

# ==========================================
# DASHBOARD GRID LAYOUT
# ==========================================
st.title("🦅 COMMAND CENTER - 5 ASSETS")

state = load_state()
markets = state.get('markets', {})

# --- RIGA 1: KPI GENERALI ---
k1, k2, k3, k4 = st.columns(4)
if state:
    k1.metric("💰 Equity", f"${state.get('equity', 0):.2f}", delta=f"{state.get('pnl_total', 0):.2f}")
    k2.metric("📉 Drawdown", f"{state.get('drawdown', 0)*100:.2f}%")
    k3.metric("🔒 Margin Used", f"${state.get('margin_used', 0):.2f}")
    k4.metric("⚡ Last Action", state.get('last_action', '-'))

st.markdown("---")

# --- RIGA 2: PANORAMICA 5 CRYPTO (GRID) ---
st.subheader("📡 Situazione Mercato Tempo Reale")

# Creiamo 5 colonne dinamiche
cols = st.columns(len(SYMBOLS))

for i, sym in enumerate(SYMBOLS):
    with cols[i]:
        mkt_data = markets.get(sym, {})
        
        # Dati
        price = mkt_data.get('price', 0)
        status = mkt_data.get('state', 'OFF')
        pnl = mkt_data.get('pnl', 0)
        p1h = mkt_data.get('p1h', 0.5)
        lev = mkt_data.get('leverage', 1)
        
        # Colore Status
        color = "gray"
        if status == "LONG": color = "green"
        elif status == "SHORT": color = "red"
        
        # Card Visuale
        st.markdown(f"### {sym}")
        st.markdown(f"**:{color}[{status}]** ({lev}x)")
        st.markdown(f"Price: `${price:.2f}`")
        
        if status != "FLAT":
            st.markdown(f"PnL: **${pnl:.2f}**")
        else:
            st.markdown("Waiting...")
            
        # Barra confidenza (0 a 1)
        st.progress(max(0.0, min(1.0, float(p1h))))
        st.caption(f"AI Conf: {p1h:.2f}")

st.markdown("---")

# --- RIGA 3: DETTAGLIO GRAFICO (Selezionabile) ---
st.subheader("🔍 Analisi Tecnica Dettagliata")

# Qui lasciamo il selettore SOLO per il grafico (altrimenti esplode la pagina con 5 grafici pesanti)
selected_symbol = st.radio("Seleziona Grafico:", SYMBOLS, horizontal=True)

history = load_history()
trades = load_trades()

if not history.empty and selected_symbol in history['symbol'].unique():
    df_sym = history[history['symbol'] == selected_symbol].copy()
    
    # Filtra Trades
    df_trades = pd.DataFrame()
    if not trades.empty:
        df_trades = trades[trades['symbol'] == selected_symbol]

    # Plot
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.03, row_heights=[0.7, 0.3],
                        subplot_titles=(f"{selected_symbol} Price", "AI Signal Strength"))

    # Candela/Linea Prezzo
    fig.add_trace(go.Scatter(x=df_sym['timestamp'], y=df_sym['price'], 
                             line=dict(color='white', width=1), name='Price'), row=1, col=1)

    # Trade Markers
    if not df_trades.empty:
        longs = df_trades[df_trades['side'] == 'LONG']
        shorts = df_trades[df_trades['side'] == 'SHORT']
        fig.add_trace(go.Scatter(x=longs['entry_time'], y=longs['entry'], mode='markers',
                                 marker=dict(symbol='triangle-up', color='#00ff00', size=14), name='Long'), row=1, col=1)
        fig.add_trace(go.Scatter(x=shorts['entry_time'], y=shorts['entry'], mode='markers',
                                 marker=dict(symbol='triangle-down', color='#ff0000', size=14), name='Short'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_trades['exit_time'], y=df_trades['exit'], mode='markers',
                                 marker=dict(symbol='x', color='orange', size=10), name='Exit'), row=1, col=1)

    # AI Lines
    fig.add_trace(go.Scatter(x=df_sym['timestamp'], y=df_sym['pred_1h'], 
                             line=dict(color='cyan', width=2), name='1H Signal'), row=2, col=1)
    
    # Zone Trigger
    fig.add_hrect(y0=0.55, y1=1.0, fillcolor="green", opacity=0.1, layer="below", row=2, col=1)
    fig.add_hrect(y0=0.0, y1=0.45, fillcolor="red", opacity=0.1, layer="below", row=2, col=1)
    fig.add_hline(y=0.5, line_dash="dot", line_color="gray", row=2, col=1)

    fig.update_layout(height=600, template="plotly_dark", margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)

# Footer Refresh
time.sleep(10) # Refresh ogni 10s
st.rerun()