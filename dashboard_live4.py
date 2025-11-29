import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
import time
from datetime import timedelta

# ==========================================
# CONFIGURAZIONE
# ==========================================
st.set_page_config(
    page_title="🦅 COMMAND CENTER - ULTIMATE",
    layout="wide",
    page_icon="🦅",
    initial_sidebar_state="collapsed"
)

# LISTA ASSET
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "AVAXUSDT", "XRPUSDT"]

# FILE
HISTORY_FILE = "history_live.csv"
TRADE_FILE = "papertrades_live_meta.csv"
STATE_FILE = "papertrade_state.json"

# Auto-refresh
if 'last_update' not in st.session_state:
    st.session_state.last_update = time.time()

def refresh():
    st.session_state.last_update = time.time()

# ==========================================
# LOADERS (Crash-Proof)
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
        if df['timestamp'].dt.tz is None:
            df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
        return df
    except Exception as e: 
        st.error(f"Errore lettura CSV: {e}")
        return pd.DataFrame()

def load_trades():
    if not os.path.exists(TRADE_FILE): return pd.DataFrame()
    try:
        df = pd.read_csv(TRADE_FILE)
        df['entry_time'] = pd.to_datetime(df['entry_time'])
        df['exit_time'] = pd.to_datetime(df['exit_time'])
        if df['entry_time'].dt.tz is None: df['entry_time'] = df['entry_time'].dt.tz_localize('UTC')
        if df['exit_time'].dt.tz is None: df['exit_time'] = df['exit_time'].dt.tz_localize('UTC')
        return df
    except: return pd.DataFrame()

# ==========================================
# LAYOUT PRINCIPALE
# ==========================================
st.title("🦅 COMMAND CENTER - ULTIMATE")

state = load_state()
markets = state.get('markets', {})

# --- 1. KPI GENERALI ---
k1, k2, k3, k4, k5 = st.columns(5)
if state:
    equity = state.get('equity', 0)
    pnl_total = state.get('pnl_total', 0)
    win_rate = state.get('winrate', 0.0) * 100
    
    k1.metric("💰 Equity", f"${equity:,.2f}", delta=f"{pnl_total:,.2f}")
    k2.metric("🎯 Win Rate", f"{win_rate:.1f}%") 
    k3.metric("📉 Drawdown", f"{state.get('drawdown', 0)*100:.2f}%")
    k4.metric("🔒 Margin Used", f"${state.get('margin_used', 0):.2f}")
    k5.metric("⚡ Last Action", state.get('last_action', '-'))

st.markdown("---")

# --- 2. GRID VIEW (PANORAMICA 5 ASSET) ---
st.subheader("📡 Situazione Mercato Tempo Reale")
cols = st.columns(len(SYMBOLS))

for i, sym in enumerate(SYMBOLS):
    with cols[i]:
        mkt_data = markets.get(sym, {})
        price = mkt_data.get('price', 0)
        status = mkt_data.get('state', 'OFF')
        pnl = mkt_data.get('pnl', 0)
        p1h = mkt_data.get('p1h', 0.5)
        lev = mkt_data.get('leverage', 1)
        
        color = "gray"
        if status == "LONG": color = "green"
        elif status == "SHORT": color = "red"
        
        st.markdown(f"### {sym}")
        st.markdown(f"**:{color}[{status}]** ({lev}x)")
        st.markdown(f"${price:.4f}")
        
        if status != "FLAT":
            pnl_color = "green" if pnl >= 0 else "red"
            st.markdown(f"PnL: **:{pnl_color}[${pnl:.2f}]**")
        else:
            st.markdown("Waiting...")
            
        st.progress(max(0.0, min(1.0, float(p1h))))
        st.caption(f"1H AI Conf: {p1h:.2f}")

st.markdown("---")

# --- 3. DEEP DIVE (ANALISI DETTAGLIATA) ---
st.subheader("🧠 Deep Dive & Neural Scan")

history = load_history()
if not history.empty:
    # Selettore Asset
    selected_symbol = st.radio("Seleziona Asset da Analizzare:", SYMBOLS, horizontal=True)
    
    # Filtra dati
    df_sym = history[history['symbol'] == selected_symbol].copy()
    
    if not df_sym.empty:
        last_row = df_sym.iloc[-1]
        
        # A. I 9 GAUGE (IL CERVELLO) - CON PROTEZIONE .GET()
        st.markdown(f"#### 🧬 Neural Breakdown: {selected_symbol}")
        
        def mini_gauge(label, val):
            if pd.isna(val): val = 0.5
            col = "#00ff00" if val > 0.55 else ("#ff0000" if val < 0.45 else "#555")
            return f"""
            <div style="background:#262730; padding:10px; border-radius:5px; border-left: 5px solid {col}; margin-bottom:10px;">
                <div style="font-size:12px; color:#aaa;">{label}</div>
                <div style="font-size:20px; font-weight:bold; color:white;">{val:.2f}</div>
            </div>
            """
        
        c_d, c_h, c_m = st.columns(3)
        
        # Usa .get() per evitare KeyError
        agg_ema = last_row.get('agg_ema', 0.5)
        agg_sma = last_row.get('agg_sma', 0.5)
        agg_1h = last_row.get('agg_1h', 0.5)
        agg_15m = last_row.get('agg_15m', 0.5)

        with c_d:
            st.info("🏛️ STRATEGIC (Daily Regime)")
            st.markdown(mini_gauge("EMA XGBoost", last_row.get('raw_d_ema_xgb', 0.5)), unsafe_allow_html=True)
            st.markdown(mini_gauge("EMA LightGBM", last_row.get('raw_d_ema_lgb', 0.5)), unsafe_allow_html=True)
            st.caption(f"Aggregato Daily: {(agg_ema+agg_sma)/2:.2f}")

        with c_h:
            st.warning("⚔️ TACTICAL (1H Trend)")
            st.markdown(mini_gauge("Spot XGBoost", last_row.get('raw_1h_xgb', 0.5)), unsafe_allow_html=True)
            st.markdown(mini_gauge("Futures PRO", last_row.get('raw_1h_pro', 0.5)), unsafe_allow_html=True)
            st.caption(f"Aggregato 1H: {agg_1h:.2f}")

        with c_m:
            st.error("🔫 OPERATIONAL (15M Timing)")
            st.markdown(mini_gauge("Spot XGBoost", last_row.get('raw_15m_xgb', 0.5)), unsafe_allow_html=True)
            st.markdown(mini_gauge("Spot LightGBM", last_row.get('raw_15m_lgb', 0.5)), unsafe_allow_html=True)
            st.caption(f"Aggregato 15M: {agg_15m:.2f}")

        # B. GRAFICO INTERATTIVO (RESTYLING)
        st.markdown("#### 📉 Timeline Segnali (Ultimi 120 Min)")
        
        cutoff_time = pd.Timestamp.utcnow() - timedelta(minutes=120)
        df_chart = df_sym[df_sym['timestamp'] >= cutoff_time]
        
        # Colonne disponibili per il multiselect
        # Nascondiamo 'agg_1h' dal multiselect perché lo mettiamo fisso
        all_cols = [c for c in df_sym.columns if 'raw_' in c or 'agg_' in c]
        avail_cols = [c for c in all_cols if c != 'agg_1h'] 
        
        # Default: 15M aggregate e Futures PRO
        default_sel = ['agg_15m', 'raw_1h_pro']
        default_sel = [c for c in default_sel if c in avail_cols]
        
        show_lines = st.multiselect("Segnali Aggiuntivi da sovrapporre:", avail_cols, default=default_sel)
        
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4], vertical_spacing=0.05)
        
        # 1. Prezzo
        fig.add_trace(go.Scatter(x=df_chart['timestamp'], y=df_chart['price'], name='Price', line=dict(color='white', width=2)), row=1, col=1)
        
        # 2. AI SIGNAL STRENGTH (La linea principale - SEMPRE VISIBILE)
        # La mettiamo Ciano Elettrico e Spessa
        if 'agg_1h' in df_sym.columns:
            fig.add_trace(go.Scatter(
                x=df_chart['timestamp'], 
                y=df_chart['agg_1h'], 
                name='⚡ AI STRENGTH (1H)', 
                line=dict(color='#00FFFF', width=3) # CIANO SPESSO
            ), row=2, col=1)

        # 3. Altre Linee (Dal multiselect)
        colors = ['magenta', 'yellow', 'orange', 'purple', 'lime']
        for i, col in enumerate(show_lines):
            fig.add_trace(go.Scatter(
                x=df_chart['timestamp'], 
                y=df_chart[col], 
                name=col, 
                line=dict(width=1, color=colors[i % len(colors)], dash='dot') # Sottili e tratteggiate
            ), row=2, col=1)
            
        # Thresholds
        fig.add_hrect(y0=0.55, y1=1.0, fillcolor="green", opacity=0.1, layer="below", row=2, col=1)
        fig.add_hrect(y0=0.0, y1=0.45, fillcolor="red", opacity=0.1, layer="below", row=2, col=1)
        fig.add_hline(y=0.5, line_color='gray', line_dash='dot', row=2, col=1)
        
        fig.update_layout(height=600, template="plotly_dark", margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("Nessun dato per questo simbolo.")

else:
    st.info("⏳ In attesa di dati storici dal bot...")

st.markdown("---")

# --- 4. REGISTRO OPERATIVO ---
st.subheader("📋 Registro Operativo")

trades = load_trades()
tab1, tab2 = st.tabs(["🟢 Posizioni Aperte", "📜 Storico Operazioni"])

with tab1:
    open_data = []
    for sym, data in markets.items():
        if data.get('state') in ['LONG', 'SHORT']:
            open_data.append({
                "Asset": sym,
                "Dir": data.get('state'),
                "Lev": f"{data.get('leverage')}x",
                "Margin": f"${data.get('margin', 0):.2f}",
                "Price": f"${data.get('price', 0):.4f}",
                "PnL": f"${data.get('pnl', 0):.2f}"
            })
    if open_data:
        st.dataframe(pd.DataFrame(open_data), use_container_width=True)
    else:
        st.info("Nessuna posizione aperta.")

with tab2:
    if not trades.empty:
        df_show = trades[['entry_time', 'symbol', 'side', 'entry', 'exit', 'pnl', 'minutes', 'reason', 'notional']].sort_values('entry_time', ascending=False)
        st.dataframe(
            df_show.style.format({
                'entry': '{:.4f}', 'exit': '{:.4f}', 'pnl': '{:.2f}', 'notional': '{:.2f}'
            }).applymap(lambda x: 'color: #00ff00' if x > 0 else 'color: #ff4444', subset=['pnl']),
            use_container_width=True
        )
    else:
        st.info("Nessun trade storico.")

# Footer refresh
time.sleep(10)
st.rerun()