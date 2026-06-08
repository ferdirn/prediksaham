"""
Batch Next-Day Prediction
==========================
Runs lstm_predictor for every ticker that has a saved config in lstm_configs.json,
collects all forecasts, and prints a ranked summary table.

Usage:
    python lstm_batch_predict.py
"""

import argparse
import json
import os
import sqlite3
import warnings
from datetime import date
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from lstm_predictor import add_indicators, inverse_close

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential

SQLITE_PATH    = "bei_stocks.db"
TICKER_CONFIGS = "lstm_configs.json"


# ── model helpers (mirrors lstm_predictor.py) ─────────────────────────────────

def load_configs() -> dict:
    path = Path(TICKER_CONFIGS)
    if not path.exists():
        print(f"  Tidak ada config di '{TICKER_CONFIGS}'. Jalankan lstm_batch_config_search.py dulu.")
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as e:
        print(f"  Config '{TICKER_CONFIGS}' corrupt atau tidak valid: {e}")
        return {}


def load_data(ticker: str, features: list) -> pd.DataFrame:
    conn = sqlite3.connect(SQLITE_PATH)
    df = pd.read_sql_query(
        "SELECT Date, Close, Open, High, Low, Volume, GainLoss_Pct, DayReturn_Pct "
        "FROM daily_prices WHERE Ticker = ? ORDER BY Date ASC",
        conn, params=(ticker.upper(),)
    )
    conn.close()
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    df = add_indicators(df)
    df.dropna(subset=features, inplace=True)
    return df


def make_sequences(data: np.ndarray, lookback: int):
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i : i + lookback])
        y.append(data[i + lookback, 0])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def build_model(cfg: dict, lookback: int) -> Sequential:
    n_features = len(cfg["features"])
    model = Sequential()
    for i in range(cfg["num_layers"]):
        rs = (i < cfg["num_layers"] - 1)
        kw = dict(return_sequences=rs)
        if i == 0:
            kw["input_shape"] = (lookback, n_features)
        model.add(LSTM(cfg["lstm_units"], **kw))
        model.add(Dropout(cfg["dropout"]))
    model.add(Dense(cfg["dense_units"], activation="relu"))
    model.add(Dense(1))
    model.compile(optimizer=tf.keras.optimizers.Adam(cfg["learning_rate"]), loss="mse")
    return model


def predict_ticker(ticker: str, cfg: dict) -> dict | None:
    np.random.seed(cfg["seed"])
    tf.random.set_seed(cfg["seed"])

    features  = cfg["features"]
    lookback  = cfg["lookback"]
    split     = cfg["train_split"]

    try:
        df = load_data(ticker, features)
    except Exception as e:
        print(f"  [{ticker}] ERROR: {e}")
        return None

    if len(df) < lookback + 10:
        print(f"  [{ticker}] SKIP: not enough data")
        return None

    raw   = df[features].values.astype(np.float32)
    idx   = int(len(raw) * split)

    scaler       = MinMaxScaler()
    train_scaled = scaler.fit_transform(raw[:idx])
    test_full    = scaler.transform(np.concatenate([raw[idx - lookback:idx], raw[idx:]], axis=0))

    X_train, y_train = make_sequences(train_scaled, lookback)
    X_test,  y_test  = make_sequences(test_full,    lookback)

    model = build_model(cfg, lookback)
    model.fit(
        X_train, y_train,
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        validation_split=0.1,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=cfg["patience"],
                          restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=7,
                              min_lr=1e-6, verbose=0),
        ],
        verbose=0,
    )

    # Test MAPE — use model() directly to avoid shared predict tf.function retracing
    test_pred  = inverse_close(scaler, model(tf.constant(X_test), training=False).numpy().flatten())
    test_actual = inverse_close(scaler, y_test)
    mape = float(np.mean(np.abs((test_actual - test_pred) / test_actual)) * 100)

    # Next-day forecast
    last_seq    = raw[-lookback:]
    last_scaled = scaler.transform(last_seq).reshape(1, lookback, len(features))
    next_scaled = model(tf.constant(last_scaled), training=False).numpy()[0][0]
    next_price  = inverse_close(scaler, [next_scaled])[0]

    last_close  = df["Close"].iloc[-1]
    last_date   = df.index[-1].date()
    change_idr  = next_price - last_close
    change_pct  = change_idr / last_close * 100
    direction   = "▲" if next_price >= last_close else "▼"

    return {
        "ticker"     : ticker,
        "last_date"  : str(last_date),
        "last_close" : round(last_close, 0),
        "forecast"   : round(next_price, 0),
        "change_idr" : round(change_idr, 0),
        "change_pct" : round(change_pct, 2),
        "direction"  : direction,
        "mape"       : round(mape, 2),
        "lookback"   : lookback,
    }


# ── main ──────────────────────────────────────────────────────────────────────

def run():
    configs = load_configs()
    results = []

    print(f"\n{'═'*60}")
    print(f"  BEI Batch Prediction — {date.today()}")
    print(f"  Tickers: {list(configs.keys())}")
    print(f"{'═'*60}\n")

    for ticker in configs:
        print(f"  Training {ticker} (lookback={configs[ticker]['lookback']})...", end=" ", flush=True)
        result = predict_ticker(ticker, configs[ticker])
        if result:
            results.append(result)
            print(f"IDR {result['last_close']:,.0f} → {result['direction']} IDR {result['forecast']:,.0f} "
                  f"({result['change_pct']:+.2f}%)  [MAPE {result['mape']:.2f}%]")
        else:
            print("FAILED")

    # Clear TF session once after all tickers — not inside the loop, to avoid
    # forcing a retrace on every subsequent model creation.
    tf.keras.backend.clear_session()

    if not results:
        print("No results.")
        return

    df = pd.DataFrame(results).sort_values("mape", ascending=True)

    print(f"\n{'═'*60}")
    print(f"  HASIL PREDIKSI BESOK — {date.today()}")
    print(f"{'═'*60}")
    print(f"\n  {'Ticker':<6}  {'Last Close':>11}  {'Forecast':>10}  {'Change':>8}  {'Change%':>8}  {'MAPE':>7}")
    print(f"  {'─'*6}  {'─'*11}  {'─'*10}  {'─'*8}  {'─'*8}  {'─'*7}")
    for _, r in df.iterrows():
        arrow = "▲" if r["change_pct"] > 0 else "▼"
        print(f"  {r['ticker']:<6}  {r['last_close']:>11,.0f}  {r['forecast']:>10,.0f}  "
              f"{r['change_idr']:>+8,.0f}  {arrow}{abs(r['change_pct']):>7.2f}%  {r['mape']:>6.2f}%")
    print(f"\n  Diurutkan: MAPE terkecil → terbesar (akurasi model tertinggi di atas)")
    print(f"  MAPE = akurasi model pada data test historis")
    print(f"{'═'*60}\n")

    # Save to CSV
    out = f"predictions_{date.today()}.csv"
    df.to_csv(out, index=False)
    print(f"  Hasil disimpan → {out}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="lstm_batch_predict.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "BEI Batch Next-Day Prediction\n"
            "==============================\n"
            "Melatih model LSTM untuk setiap ticker yang memiliki konfigurasi\n"
            "tersimpan di lstm_configs.json, lalu memprediksi harga penutupan\n"
            "hari berikutnya. Hasil ditampilkan sebagai tabel ranking dan\n"
            "disimpan ke file CSV."
        ),
        epilog=(
            "Prasyarat:\n"
            "  - bei_stocks.db harus berisi data untuk ticker yang ingin diprediksi\n"
            "  - lstm_configs.json harus memiliki konfigurasi optimal per ticker\n"
            "\n"
            "Alur kerja yang disarankan:\n"
            "  1. Download data    : python bei_stock_downloader.py --file watchlist.txt --years 5\n"
            "  2. Cari config      : python lstm_batch_config_search.py\n"
            "  3. Jalankan prediksi: python lstm_batch_predict.py\n"
            "\n"
            "Output:\n"
            "  - Tabel prediksi di terminal (diurutkan dari potensi kenaikan tertinggi)\n"
            "  - predictions_YYYY-MM-DD.csv berisi semua hasil prediksi\n"
            "\n"
            "Contoh:\n"
            "  python lstm_batch_predict.py\n"
        ),
    )
    parser.parse_args()
    run()
