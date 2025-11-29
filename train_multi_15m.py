import os
import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, roc_curve
from sklearn.model_selection import train_test_split

from features import add_features_pro
import joblib

DATA_DIR = "data_1m"
MODEL_DIR = "modelli"
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "AVAXUSDT", "XRPUSDT"]

os.makedirs(MODEL_DIR, exist_ok=True)

# ===========================
# k ottimali per simbolo
# (da k_sweep_15m.csv)
# ===========================
K_MAP = {
    "BTCUSDT": 0.6,
    "ETHUSDT": 0.6,
    "SOLUSDT": 1.0,
    "AVAXUSDT": 0.6,
    "XRPUSDT": 0.6,  # volendo puoi escluderla dal training
}


# ============================================================
# Soglia ottimale via Youden
# ============================================================
def optimal_threshold(y_true, y_prob):
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    j = tpr - fpr
    idx = np.argmax(j)
    return thr[idx]


# ============================================================
# TRAINING COMPLETO
# ============================================================
def train_for_symbol(symbol):
    csv_path = os.path.join(DATA_DIR, f"{symbol}_1m.csv")
    if not os.path.exists(csv_path):
        print(f"[{symbol}] CSV non trovato: {csv_path}")
        return

    print("\n=======================================")
    print(f"  TRAIN MULTICLASS DIREZIONALE: {symbol}")
    print("=======================================\n")

    # --------------------------------------------------------
    # CARICA RAW
    # --------------------------------------------------------
    df_raw = pd.read_csv(csv_path)

    # k specifico per simbolo (fallback 0.8)
    k = K_MAP.get(symbol, 0.8)

    # --------------------------------------------------------
    # FEATURE ENGINEERING + TARGET DIREZIONALE (+1/-1/0)
    # --------------------------------------------------------
    df = add_features_pro(df_raw, horizon=15, k=k)

    if len(df) < 5000:
        print(f"[{symbol}] Troppi pochi dati dopo feature: {len(df)}")
        return

    # --------------------------------------------------------
    # FILTRO ORARIO (12–21 UTC)
    # --------------------------------------------------------
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["hour"] = df["timestamp"].dt.hour
        df = df[df["hour"].between(12, 21)]

    # --------------------------------------------------------
    # RIMUOVI target = 0 (no movimento forte)
    # --------------------------------------------------------
    df = df[df["target"] != 0]

    if len(df) < 2000:
        print(f"[{symbol}] Dati insufficienti dopo filtro target!=0: {len(df)}")
        return

    # --------------------------------------------------------
    # MAPPA target in binario (+1 -> 1, -1 -> 0)
    # --------------------------------------------------------
    df["target_bin"] = (df["target"] == 1).astype(int)

    # salva mapping (informativo)
    mapping_path = os.path.join(MODEL_DIR, f"mapping_15m_{symbol}.txt")
    with open(mapping_path, "w") as f:
        f.write("target +1 => 1 (LONG)\n")
        f.write("target -1 => 0 (SHORT)\n")
        f.write(f"k usato: {k}\n")

    # --------------------------------------------------------
    # TEMPORAL DROPOUT sull'intero df (NON solo su X)
    # --------------------------------------------------------
    df = df.sample(frac=0.8, random_state=42).sort_values("timestamp")

    # --------------------------------------------------------
    # SEPARAZIONE X / y (dopo il dropout)
    # --------------------------------------------------------
    y = df["target_bin"]
    X = df.drop(columns=["target", "target_bin", "timestamp", "hour"], errors="ignore")

    print(f"[{symbol}] Dopo cleanup: X={len(X)}, y={len(y)}")

    # --------------------------------------------------------
    # SPLIT 75/25 SENZA SHUFFLE
    # --------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, shuffle=False
    )

    # ============================================================
    # BILANCIAMENTO CLASSE
    # ============================================================
    pos = y_train.sum()
    neg = len(y_train) - pos
    if pos == 0 or neg == 0:
        print(f"[{symbol}] ATTENZIONE: una sola classe nel training (pos={pos}, neg={neg}).")
        return

    scale_pos_weight = neg / (pos + 1e-9)
    print(f"[{symbol}] pos={pos}, neg={neg}, scale_pos_weight={scale_pos_weight:.2f}")

    # ============================================================
    # XGBOOST POTENZIATO
    # ============================================================
    print(f"[{symbol}] Addestro XGBoost…")

    xgb = XGBClassifier(
        n_estimators=650,
        max_depth=7,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.9,
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        objective="binary:logistic",
        eval_metric="logloss",
    )

    xgb.fit(X_train, y_train)
    p_xgb = xgb.predict_proba(X_test)[:, 1]
    pred_xgb = (p_xgb > 0.5).astype(int)
    acc_xgb = accuracy_score(y_test, pred_xgb)
    print(f"[{symbol}] XGB accuracy (thr=0.5): {acc_xgb:.4f}")

    # ============================================================
    # LIGHTGBM POTENZIATO
    # ============================================================
    print(f"[{symbol}] Addestro LightGBM…")

    lgb = LGBMClassifier(
        n_estimators=800,
        learning_rate=0.03,
        max_depth=8,
        subsample=0.9,
        colsample_bytree=0.9,
        is_unbalance=True
    )
    lgb.fit(X_train, y_train)
    p_lgb = lgb.predict_proba(X_test)[:, 1]
    pred_lgb = (p_lgb > 0.5).astype(int)
    acc_lgb = accuracy_score(y_test, pred_lgb)
    print(f"[{symbol}] LGB accuracy (thr=0.5): {acc_lgb:.4f}")

    # ============================================================
    # ENSEMBLE MAX()
    # ============================================================
    p_final = np.maximum(p_xgb, p_lgb)
    pred_final = (p_final > 0.5).astype(int)
    acc_final = accuracy_score(y_test, pred_final)
    print(f"[{symbol}] ENSEMBLE accuracy (thr=0.5): {acc_final:.4f}")

    # ============================================================
    # SOGLIA OTTIMALE
    # ============================================================
    thr = optimal_threshold(y_test, p_final)
    print(f"[{symbol}] Soglia ottimale ensemble: {thr:.3f}")

    thr_path = os.path.join(MODEL_DIR, f"thr_15m_{symbol}.txt")
    with open(thr_path, "w") as f:
        f.write(str(thr))

    # ============================================================
    # SALVA MODELLI
    # ============================================================
    joblib.dump(xgb, os.path.join(MODEL_DIR, f"xgb_15m_{symbol}.pkl"))
    joblib.dump(lgb, os.path.join(MODEL_DIR, f"lgb_15m_{symbol}.pkl"))
    print(f"[{symbol}] Modelli salvati in {MODEL_DIR}/")

    # ============================================================
    # REPORT FINALE
    # ============================================================
    print("\nEsempi previsioni (prime 15 righe test):")
    for i in range(min(15, len(y_test))):
        print(f" idx={i} | true={y_test.iloc[i]} | p_final={p_final[i]:.3f}")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    for sym in SYMBOLS:
        train_for_symbol(sym)
