import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import time

STATE_FILE = "papertrade_state.json"

def load_state():
    """Carica lo stato corrente dal file JSON."""
    if not os.path.exists(STATE_FILE):
        st.warning(f"File di stato non trovato: {STATE_FILE}. Assicurati che 'papertrade_live.py' sia in esecuzione.")
        return None
    try:
        with open(STATE_FILE, "r") as f:
            data = json.load(f)
        return data
    except Exception as e:
        st.error(f"Errore durante la lettura dello stato: {e}. Controlla il formato JSON.")
        return None

def main():
    # Configurazione iniziale della pagina
    st.set_page_config(
        page_title="Meta-Coordinator Live Dashboard",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.title("📈 Meta-Coordinator Live Dashboard (Paper Trading)")

    # Aggiornamento automatico (loop infinito)
    st.header("Aggiornamento Live")
    placeholder = st.empty()

    while True:
        data = load_state()

        with placeholder.container():
            if data is None:
                st.info("In attesa dei dati... (Avvia 'papertrade_live.py' per generare il file di stato)")
                time.sleep(5)
                continue

            # === 0. Intestazione ===
            last_update = pd.to_datetime(data.get("timestamp", "N/A")).strftime("%Y-%m-%d %H:%M:%S")
            st.markdown(f"Ultimo aggiornamento: **{last_update} (UTC)** | Ultima Azione: **{data.get('last_action', 'NOOP')}**")

            # === 1. Statistiche Globali ===
            st.markdown("---")
            st.header("1. Performance di Conto")

            col1, col2, col3, col4, col5 = st.columns(5)

            col1.metric("Equity Corrente", f"€{data['equity']:.2f}")
            col2.metric("P&L Totale", f"€{data['pnl_total']:.2f}")
            col3.metric("Margine Usato", f"€{data['margin_used']:.2f}")
            col4.metric("Drawdown Max", f"{data['drawdown'] * 100:.2f}%")
            col5.metric("Win Rate", f"{data['winrate'] * 100:.1f}% ({data['trades_closed']})")

            # === 2. Snapshot di Mercato (Posizioni e Segnali) ===
            st.markdown("---")
            st.header("2. Snapshot di Mercato e Segnali")

            market_data = data.get("markets", {})
            if market_data:
                df_markets = pd.DataFrame.from_dict(market_data, orient="index")
                df_markets.index.name = "Symbol"

                # Rinomina colonne per chiarezza
                df_markets.columns = [
                    "Prezzo", "Prob. 15M", "Prob. 1H", "Prob. EMA50_1D", "Prob. SMA200_1D",
                    "Stato Posizione", "Minuti Apertura", "P&L Non Realizzato (€)"
                ]

                # Funzioni di stile per colorare probabilità e stato
                def color_prob(val):
                    """Colora la probabilità in base alla distanza da 0.5 (Verde per LONG, Rosso per SHORT)."""
                    if isinstance(val, (float, np.floating)) and 0.5 < val < 1.0:
                        intensity = int((val - 0.5) * 2 * 255)
                        return f'background-color: rgba(0, 255, 0, {intensity/255*0.3})' # Verde chiaro
                    elif isinstance(val, (float, np.floating)) and 0.0 < val < 0.5:
                        intensity = int((0.5 - val) * 2 * 255)
                        return f'background-color: rgba(255, 0, 0, {intensity/255*0.3})' # Rosso chiaro
                    return ''

                def color_state(val):
                    """Colora lo stato della posizione."""
                    if val == "LONG":
                        return 'background-color: #d4edda; color: #155724' # Verde
                    elif val == "SHORT":
                        return 'background-color: #f8d7da; color: #721c24' # Rosso
                    return 'background-color: #fdfdfe; color: #6c757d' # Grigio

                st.dataframe(
                    df_markets.style.format(
                        {
                            "Prezzo": "{:.4f}",
                            "Prob. 15M": "{:.3f}",
                            "Prob. 1H": "{:.3f}",
                            "Prob. EMA50_1D": "{:.3f}",
                            "Prob. SMA200_1D": "{:.3f}",
                            "P&L Non Realizzato (€)": "{:,.2f}",
                        }
                    ).applymap(color_prob, subset=[
                        "Prob. 15M", "Prob. 1H", "Prob. EMA50_1D", "Prob. SMA200_1D"
                    ]).applymap(color_state, subset=["Stato Posizione"]),
                    use_container_width=True
                )
            else:
                st.info("Nessun dato di mercato disponibile.")


            # === 3. Ultimi Trade Chiusi ===
            st.markdown("---")
            st.header("3. Ultimi Trade Chiusi (Max 20)")

            trades = data.get("trades", [])
            if trades:
                df_trades = pd.DataFrame(trades)
                df_trades.columns = [
                    "Symbol", "Side", "Entry Price", "Exit Price",
                    "P&L (€)", "Durata (min)"
                ]

                # Stile per colorare il P&L
                def color_pnl(val):
                    color = 'red' if val < 0 else 'green' if val > 0 else 'grey'
                    return f'color: {color}'

                st.dataframe(
                    df_trades.style.format(
                        {
                            "Entry Price": "{:.4f}",
                            "Exit Price": "{:.4f}",
                            "P&L (€)": "{:,.2f}",
                        }
                    ).applymap(color_pnl, subset=["P&L (€)"]),
                    use_container_width=True
                )
            else:
                st.info("Nessun trade chiuso di recente.")


        # Pausa per l'aggiornamento (5 secondi)
        time.sleep(5)

if __name__ == '__main__':
    main()