# decision_engine.py
#
# Combina modello 15m (XGB+LGB) e modello 1h (XGB)
# per produrre una decisione operativa multilivello.

import os
import sys
import pandas as pd
import joblib

from features_15m import add_features_pro
from features_1h import resample_1m_to_1h, add_features_1h

DATA_DIR = "data_1m"
MODEL_DIR = "modelli"

# Soglie per 15m
P15_LONG_STRONG  = 0.70   # sopra → LONG forte
P15_LONG_NORMAL  = 0.62   # sopra → LONG normale
P15_SHORT_STRONG = 0.30   # sotto → SHORT forte
P15_SHORT_NORMAL = 0.38   # sotto → SHORT normale

# Soglie per 1h (trend di contesto)
P1H_TREND_UP   = 0.60
P1H_TREND_DOWN = 0.40


def load_models_15m(symbol):
    xgb_path = os.path.join(MODEL_DIR, f"xgb_15m_{symbol}.pkl")
    lgb_path = os.path.join(MODEL_DIR, f"lgb_15m_{symbol}.pkl")

    if not os.path.exists(xgb_path):
        raise FileNotFoundError(f"Modello XGB 15m non trovato: {xgb_path}")
    if not os.path.exists(lgb_path):
        raise FileNotFoundError(f"Modello LGB 15m non trovato: {lgb_path}")

    xgb = joblib.load(xgb_path)
    lgb = joblib.load(lgb_path)
    return xgb, lgb


def load_model_1h(symbol):
    path = os.path.join(MODEL_DIR, f"xgb_1h_{symbol}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Modello XGB 1h non trovato: {path}")
    return joblib.load(path)


def compute_p15(symbol):
    """
    Ritorna p15 = prob. ensemble LONG su orizzonte 15m per l'ultima riga.
    """
    csv_path = os.path.join(DATA_DIR, f"{symbol}_1m.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV 1m non trovato: {csv_path}")

    df_raw = pd.read_csv(csv_path)
    df_feat = add_features_pro(df_raw)  # funzione già usata per il training 15m

    if len(df_feat) == 0:
        raise RuntimeError(f"Nessuna riga di feature 15m per {symbol}")

    X = df_feat.drop(columns=["target"])
    x_last = X.iloc[-1:].copy()

    xgb, lgb = load_models_15m(symbol)

    p_xgb = xgb.predict_proba(x_last)[0, 1]
    p_lgb = lgb.predict_proba(x_last)[0, 1]

    p_ens = 0.5 * p_xgb + 0.5 * p_lgb
    return p_ens, p_xgb, p_lgb


def compute_p1h(symbol):
    """
    Ritorna p1h = prob. LONG oraria (next 1h > fee) per l'ultima riga 1h.
    """
    csv_path = os.path.join(DATA_DIR, f"{symbol}_1m.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV 1m non trovato: {csv_path}")

    df_1m = pd.read_csv(csv_path)
    df_1h = resample_1m_to_1h(df_1m)
    df_feat_1h = add_features_1h(df_1h, fee=0.0005, horizon=1)

    if len(df_feat_1h) == 0:
        raise RuntimeError(f"Nessuna riga di feature 1h per {symbol}")

    X_1h = df_feat_1h.drop(columns=["target"])
    x_last_1h = X_1h.iloc[-1:].copy()

    model_1h = load_model_1h(symbol)
    p1h = model_1h.predict_proba(x_last_1h)[0, 1]

    return p1h


def decide(symbol):
    """
    Combina p15 e p1h e stampa una decisione.
    """

    p15, p15_xgb, p15_lgb = compute_p15(symbol)
    p1h = compute_p1h(symbol)

    # Determina “direzione 15m”
    if p15 >= P15_LONG_STRONG:
        dir_15m = "LONG_STRONG"
    elif p15 >= P15_LONG_NORMAL:
        dir_15m = "LONG_NORMAL"
    elif p15 <= P15_SHORT_STRONG:
        dir_15m = "SHORT_STRONG"
    elif p15 <= P15_SHORT_NORMAL:
        dir_15m = "SHORT_NORMAL"
    else:
        dir_15m = "NO_TRADE"

    # Determina “trend 1h”
    if p1h >= P1H_TREND_UP:
        trend_1h = "UP"
    elif p1h <= P1H_TREND_DOWN:
        trend_1h = "DOWN"
    else:
        trend_1h = "NEUTRAL"

    # Log di base
    print("============================================")
    print(f"Symbol: {symbol}")
    print("--------------------------------------------")
    print(f"15m ensemble prob_up  : {p15:.3f}")
    print(f"  - XGB 15m prob_up   : {p15_xgb:.3f}")
    print(f"  - LGB 15m prob_up   : {p15_lgb:.3f}")
    print(f"1h  prob_up           : {p1h:.3f}")
    print("--------------------------------------------")
    print(f"Direzione 15m (raw)   : {dir_15m}")
    print(f"Trend 1h              : {trend_1h}")
    print("--------------------------------------------")

    # Decisione finale
    decision = "NO_TRADE"
    size_factor = 0.0  # 0 = niente, 1 = size piena

    # CASI LONG
    if dir_15m in ("LONG_STRONG", "LONG_NORMAL"):
        if trend_1h == "UP":
            decision = "ENTER_LONG_STRONG"
            size_factor = 1.0 if dir_15m == "LONG_STRONG" else 0.7
        elif trend_1h == "NEUTRAL":
            decision = "ENTER_LONG_NORMAL"
            size_factor = 0.6 if dir_15m == "LONG_STRONG" else 0.4
        else:  # trend_1h == "DOWN"
            # long contro-trend → solo eventuale scalp leggero
            decision = "SCALP_LONG_CONTRA_TREND"
            size_factor = 0.2

    # CASI SHORT
    elif dir_15m in ("SHORT_STRONG", "SHORT_NORMAL"):
        if trend_1h == "DOWN":
            decision = "ENTER_SHORT_STRONG"
            size_factor = 1.0 if dir_15m == "SHORT_STRONG" else 0.7
        elif trend_1h == "NEUTRAL":
            decision = "ENTER_SHORT_NORMAL"
            size_factor = 0.6 if dir_15m == "SHORT_STRONG" else 0.4
        else:  # trend_1h == "UP"
            decision = "SCALP_SHORT_CONTRA_TREND"
            size_factor = 0.2

    # direzione 15m = NO_TRADE → decisione resta NO_TRADE
    print(f"DECISIONE FINALE      : {decision}")
    print(f"SUGGESTED SIZE FACTOR : {size_factor:.2f}")
    print("============================================")

    # Se vuoi usarlo da altro codice:
    return {
        "symbol": symbol,
        "p15_ens": float(p15),
        "p15_xgb": float(p15_xgb),
        "p15_lgb": float(p15_lgb),
        "p1h": float(p1h),
        "dir_15m": dir_15m,
        "trend_1h": trend_1h,
        "decision": decision,
        "size_factor": size_factor,
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python decision_engine.py BTCUSDT")
    else:
        decide(sys.argv[1])
