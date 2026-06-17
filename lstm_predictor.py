"""
BEI LSTM Stock Price Predictor
===============================
Predicts next-day closing price for BEI (IDX) stocks using a stacked LSTM model.
Reads historical data from bei_stocks.db (populated by bei_stock_downloader.py).
Saves prediction plot to prediction_images/{TICKER}_lstm_prediction.png.

Usage:
    python lstm_predictor.py --ticker BBCA
    python lstm_predictor.py --ticker BBCA --lookback 20 --epochs 100
    python lstm_predictor.py --ticker BBCA --forecast 3
    python lstm_predictor.py --ticker TLKM --lookback 22 --save-config

CLI Parameters:
    --ticker      IDX ticker code (required). Contoh: BBCA, TLKM, ANTM.
                  Config optimal akan dimuat otomatis dari lstm_configs.json
                  jika tersedia — tidak perlu set parameter lain secara manual.

    --lookback    Jumlah hari ke belakang yang dilihat model per sequence.
                  Makin panjang = model menangkap tren lebih jauh, tapi butuh
                  lebih banyak data dan lebih lambat. Optimal per ticker berbeda,
                  cari dengan lstm_config_search.py. (default: 48)

    --forecast    Jumlah hari ke depan yang diprediksi. forecast=1 = besok saja,
                  forecast=5 = 5 hari ke depan sekaligus. (default: 1)

    --epochs      Batas maksimum epoch training. Early stopping akan menghentikan
                  lebih awal jika val_loss tidak membaik. (default: 300)

    --db          Path ke file SQLite. Ganti jika DB ada di lokasi lain.
                  (default: bei_stocks.db)

    --save-config Simpan semua parameter run ini ke lstm_configs.json untuk
                  ticker ini. Dipakai setelah menemukan konfigurasi optimal.

    --verbose     Tampilkan detail training: model summary, progress per epoch,
                  dan pesan dari EarlyStopping / ReduceLROnPlateau.
                  Default: tidak ditampilkan — hanya hasil akhir yang muncul.

CONFIG Constants (edit langsung di source code untuk ubah default):
    FEATURES      Kolom yang dipakai sebagai input model. Urutan pertama harus
                  selalu Close karena menjadi target prediksi.
                  Default: ["Close", "Volume", "GainLoss_Pct", "DayReturn_Pct"]

    LSTM_UNITS    Jumlah neuron per layer LSTM. Lebih besar = kapasitas lebih
                  tinggi tapi rawan overfit pada data kecil. (default: 64)

    NUM_LAYERS    Jumlah layer LSTM yang ditumpuk (stacked). (default: 2)

    DROPOUT       Fraksi neuron yang dinonaktifkan saat training untuk mencegah
                  overfit. (default: 0.2)

    DENSE_UNITS   Neuron di layer Dense sebelum output. (default: 32)

    BATCH_SIZE    Jumlah sequence per update bobot. (default: 32)

    LEARNING_RATE Kecepatan update bobot optimizer Adam. (default: 0.001)

    PATIENCE      Epoch tanpa perbaikan val_loss sebelum early stopping. (default: 25)

    TRAIN_SPLIT   Proporsi data untuk training. (default: 0.8)

    SEED          Random seed untuk reproducibility. (default: 42)
"""

import argparse
import json
import sqlite3
import sys
import threading
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless — saves plot to file instead of opening window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")
import os
import random
os.environ["TF_CPP_MIN_LOG_LEVEL"]        = "3"
os.environ["TF_DETERMINISTIC_OPS"]        = "1"   # force deterministic GPU/CPU ops
os.environ["TF_DISABLE_SEGMENT_REDUCTION_OP_DETERMINISM_EXCEPTIONS"] = "1"
os.environ["PYTHONHASHSEED"]              = "42"
random.seed(42)
np.random.seed(42)

import tensorflow as tf
tf.random.set_seed(42)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential

# ──────────────────────────────────────────────
# CONFIG — defaults (overridden by lstm_configs.json if available)
# ──────────────────────────────────────────────
SQLITE_PATH      = "bei_stocks.db"
TICKER_CONFIGS   = "lstm_configs.json"

LOOKBACK      = 48
FORECAST      = 1
TRAIN_SPLIT   = 0.8
FEATURES      = ["Close", "Volume", "GainLoss_Pct", "DayReturn_Pct"]
TARGET        = "Close"

LSTM_UNITS    = 64
NUM_LAYERS    = 2
DROPOUT       = 0.2
DENSE_UNITS   = 32

EPOCHS        = 300
BATCH_SIZE    = 32
LEARNING_RATE = 0.001
PATIENCE      = 25
SEED          = 42


class Spinner:
    """Rotating spinner shown in the terminal while training runs in the foreground."""

    FRAMES = ["|", "/", "─", "\\"]

    def __init__(self, message: str = "  Training"):
        self.message = message
        self._stop  = threading.Event()
        self._thread = threading.Thread(target=self._spin, daemon=True)

    def _spin(self):
        i = 0
        while not self._stop.is_set():
            frame = self.FRAMES[i % len(self.FRAMES)]
            sys.stdout.write(f"\r  {self.message}  {frame}")
            sys.stdout.flush()
            time.sleep(0.1)
            i += 1

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join()
        sys.stdout.write("\r" + " " * (len(self.message) + 10) + "\r")
        sys.stdout.flush()


def load_ticker_config(ticker: str) -> dict:
    """Load saved config for a ticker from lstm_configs.json, if it exists."""
    path = Path(TICKER_CONFIGS)
    if not path.exists():
        return {}
    configs = json.loads(path.read_text())
    cfg = configs.get(ticker.upper(), {})
    if cfg:
        print(f"  Config loaded from {TICKER_CONFIGS} for {ticker.upper()}")
    return cfg


def save_ticker_config(ticker: str, cfg: dict):
    """Upsert a ticker config into lstm_configs.json."""
    path = Path(TICKER_CONFIGS)
    configs = json.loads(path.read_text()) if path.exists() else {}
    configs[ticker.upper()] = cfg
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(configs, indent=2))
    tmp.replace(path)
    print(f"  Config saved to {TICKER_CONFIGS} for {ticker.upper()}")


# ──────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────

def load_data(ticker: str, db_path: str = SQLITE_PATH,
              features: list[str] = None) -> pd.DataFrame:
    features = features or FEATURES
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        "SELECT Date, Close, Open, High, Low, Volume, GainLoss_Pct, DayReturn_Pct "
        "FROM daily_prices WHERE Ticker = ? ORDER BY Date ASC",
        conn, params=(ticker.upper().strip(),)
    )
    conn.close()

    if df.empty:
        raise ValueError(f"No data found for ticker '{ticker}'. Run bei_stock_downloader.py first.")

    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)

    df = add_indicators(df)

    # Drop NaN rows produced by rolling windows (first ~20 rows)
    df.dropna(subset=features, inplace=True)

    print(f"  Loaded {len(df)} rows for {ticker}  ({df.index[0].date()} → {df.index[-1].date()})")
    return df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close, high, low = df["Close"], df["High"], df["Low"]

    # Moving averages
    df["MA5"]  = close.rolling(5).mean()
    df["MA20"] = close.rolling(20).mean()

    # RSI(14)
    delta     = close.diff()
    gain      = delta.clip(lower=0).rolling(14).mean()
    loss      = (-delta.clip(upper=0)).rolling(14).mean()
    df["RSI14"] = 100 - (100 / (1 + gain / loss.replace(0, 1e-9)))

    # ATR(14) — Average True Range
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    df["ATR14"] = tr.rolling(14).mean()

    return df


# ──────────────────────────────────────────────
# PREPROCESSING
# ──────────────────────────────────────────────

def make_sequences(data: np.ndarray, lookback: int, forecast: int):
    """
    Slide a window of `lookback` steps over `data` and produce X, y pairs.
    X shape: (samples, lookback, features)
    y shape: (samples, forecast)
    """
    X, y = [], []
    for i in range(len(data) - lookback - forecast + 1):
        X.append(data[i : i + lookback])
        y.append(data[i + lookback : i + lookback + forecast, 0])  # col 0 = Close (target)
    return np.array(X), np.array(y)


def preprocess(df: pd.DataFrame, lookback: int, forecast: int, train_split: float,
               train_all: bool = False, features: list[str] = None):
    feature_data = df[features or FEATURES].values.astype(np.float32)
    dates_all    = df.index.to_numpy()
    split_idx    = int(len(feature_data) * (1.0 - train_split if train_all else train_split))

    if train_all:
        # Train on ALL data; test on last (1 - train_split) portion
        # Scaler fit on all data since model sees it all during training
        scaler       = MinMaxScaler()
        all_scaled   = scaler.fit_transform(feature_data)
        X_train, y_train = make_sequences(all_scaled, lookback, forecast)

        # Test sequences: last portion (shares rows with train — intentional)
        test_raw     = np.concatenate([feature_data[split_idx - lookback:]], axis=0)
        test_scaled  = scaler.transform(test_raw)
        X_test,  y_test  = make_sequences(test_scaled, lookback, forecast)

        train_dates  = dates_all[lookback : len(feature_data)]
        test_dates   = dates_all[split_idx : split_idx + len(y_test)]
    else:
        # Standard split: train on first 80%, test on last 20% (no overlap)
        train_raw = feature_data[:split_idx]
        test_raw  = feature_data[split_idx:]

        scaler       = MinMaxScaler()
        train_scaled = scaler.fit_transform(train_raw)

        # Include lookback tail of train so first test sequences are complete
        test_full    = np.concatenate([train_raw[-lookback:], test_raw], axis=0)
        test_scaled  = scaler.transform(test_full)

        X_train, y_train = make_sequences(train_scaled, lookback, forecast)
        X_test,  y_test  = make_sequences(test_scaled,  lookback, forecast)

        train_dates  = dates_all[lookback : split_idx]
        test_dates   = dates_all[split_idx : split_idx + len(y_test)]

    return X_train, y_train, X_test, y_test, scaler, train_dates, test_dates


def inverse_close(scaler: MinMaxScaler, scaled_values) -> np.ndarray:
    """Inverse-transform only the Close column (col 0)."""
    vals = np.asarray(scaled_values).flatten()
    n_features = scaler.scale_.shape[0]
    dummy = np.zeros((len(vals), n_features), dtype=np.float32)
    dummy[:, 0] = vals
    return scaler.inverse_transform(dummy)[:, 0]


# ──────────────────────────────────────────────
# MODEL
# ──────────────────────────────────────────────

def build_model(lookback: int, n_features: int, forecast: int,
                lstm_units: int = LSTM_UNITS, num_layers: int = NUM_LAYERS,
                dropout: float = DROPOUT, dense_units: int = DENSE_UNITS,
                learning_rate: float = LEARNING_RATE) -> Sequential:
    model = Sequential(name="BEI_LSTM")

    for i in range(num_layers):
        return_seq = (i < num_layers - 1)
        if i == 0:
            model.add(LSTM(lstm_units, return_sequences=return_seq,
                           input_shape=(lookback, n_features)))
        else:
            model.add(LSTM(lstm_units, return_sequences=return_seq))
        model.add(Dropout(dropout))

    model.add(Dense(dense_units, activation="relu"))
    model.add(Dense(forecast))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"]
    )
    return model


# ──────────────────────────────────────────────
# EVALUATION
# ──────────────────────────────────────────────

def evaluate(y_true: np.ndarray, y_pred: np.ndarray, label: str = "Test"):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    print(f"\n  [{label}]  MAE={mae:,.0f} IDR   RMSE={rmse:,.0f} IDR   MAPE={mape:.2f}%")
    return {"mae": mae, "rmse": rmse, "mape": mape}


# ──────────────────────────────────────────────
# PLOT
# ──────────────────────────────────────────────

def save_plot(ticker: str, train_dates, train_actual, test_dates, test_actual, test_pred,
              history, output_dir: Path, lookback: int = LOOKBACK):
    fig, axes = plt.subplots(2, 1, figsize=(14, 9))
    fig.suptitle(f"{ticker} — LSTM Price Prediction  (lookback={lookback})", fontsize=14)

    # ── price chart ──
    ax = axes[0]
    ax.plot(train_dates, train_actual, label="Train actual", color="steelblue", linewidth=1)
    ax.plot(test_dates,  test_actual,  label="Test actual",  color="green",     linewidth=1.2)
    ax.plot(test_dates,  test_pred,    label="Test predicted", color="red",     linewidth=1.2, linestyle="--")
    ax.set_ylabel("Close (IDR)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # ── loss curve ──
    ax2 = axes[1]
    ax2.plot(history.history["loss"],     label="Train loss", color="steelblue")
    ax2.plot(history.history["val_loss"], label="Val loss",   color="orange")
    ax2.set_ylabel("MSE Loss")
    ax2.set_xlabel("Epoch")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    out_path = output_dir / f"{ticker}_lstm_prediction.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Plot saved → {out_path}")


# ──────────────────────────────────────────────
# MAIN PIPELINE
# ──────────────────────────────────────────────

def _resolve_config(ticker: str, lookback: int, forecast: int, epochs: int,
                    train_split_override: float | None) -> dict:
    """Merge CLI args with saved ticker config; CLI args act as defaults only."""
    cfg = load_ticker_config(ticker)
    return {
        "features"      : cfg.get("features",       FEATURES),
        "lookback"      : cfg.get("lookback",        lookback),
        "forecast"      : cfg.get("forecast",        forecast),
        "epochs"        : cfg.get("epochs",          epochs),
        "patience"      : cfg.get("patience",        PATIENCE),
        "seed"          : cfg.get("seed",            SEED),
        "lstm_units"    : cfg.get("lstm_units",      LSTM_UNITS),
        "num_layers"    : cfg.get("num_layers",      NUM_LAYERS),
        "dropout"       : cfg.get("dropout",         DROPOUT),
        "dense_units"   : cfg.get("dense_units",     DENSE_UNITS),
        "batch_size"    : cfg.get("batch_size",      BATCH_SIZE),
        "learning_rate" : cfg.get("learning_rate",   LEARNING_RATE),
        "train_split"   : train_split_override if train_split_override is not None
                          else cfg.get("train_split", TRAIN_SPLIT),
    }


def _forecast_next(model, scaler, df: pd.DataFrame, features: list[str],
                   lookback: int, forecast: int) -> np.ndarray:
    last_seq = df[features].values[-lookback:].astype(np.float32)
    last_seq_scaled = scaler.transform(last_seq).reshape(1, lookback, len(features))
    next_pred_scaled = model.predict(last_seq_scaled, verbose=0)
    dummy = np.zeros((forecast, scaler.scale_.shape[0]), dtype=np.float32)
    dummy[:, 0] = next_pred_scaled[0]
    return scaler.inverse_transform(dummy)[:, 0]


def _print_forecast(ticker: str, df: pd.DataFrame, next_prices: np.ndarray) -> None:
    last_close = df["Close"].iloc[-1]
    last_date  = df.index[-1].date()
    label_close = f"Last close  ({last_date})"
    col_width   = max(len(label_close),
                      max(len(f"Forecast +{i}d") for i in range(1, len(next_prices) + 1)))
    print(f"\n{'─'*56}")
    print(f"  {ticker}")
    print(f"  {label_close:<{col_width}} : IDR {last_close:,.0f}")
    for i, price in enumerate(next_prices, 1):
        change_pct = (price - last_close) / last_close * 100
        direction  = "▲" if price >= last_close else "▼"
        print(f"  {'Forecast +' + str(i) + 'd':<{col_width}} : IDR {price:,.0f}  {direction} {change_pct:+.2f}%")
    print(f"{'─'*56}\n")


def run(ticker: str, lookback: int, forecast: int, epochs: int, db_path: str,
        verbose: bool = False, train_all: bool = False,
        train_split_override: float = None):
    c = _resolve_config(ticker, lookback, forecast, epochs, train_split_override)

    mode_label = "train_all=True (seluruh data)" if train_all else f"train_split={c['train_split']}"
    print(f"\n{'═'*56}")
    print(f"  BEI LSTM Predictor — {ticker}")
    print(f"  lookback={c['lookback']}  forecast={c['forecast']}  epochs={c['epochs']}  seed={c['seed']}")
    print(f"  features={c['features']}")
    print(f"  mode={mode_label}")
    print(f"{'═'*56}\n")

    # 1. Load
    df = load_data(ticker, db_path, features=c["features"])

    # 2. Preprocess
    X_train, y_train, X_test, y_test, scaler, train_dates, test_dates = preprocess(
        df, c["lookback"], c["forecast"], c["train_split"],
        train_all=train_all, features=c["features"],
    )

    print(f"  Train sequences : {len(X_train)}")
    print(f"  Test  sequences : {len(X_test)}")

    if len(X_train) < 50:
        print("\n  WARNING: fewer than 50 training sequences. Results will be unreliable.")
        print("  Run bei_stock_downloader.py with --years 5 to get more data.\n")

    # 3. Build
    model = build_model(
        c["lookback"], len(c["features"]), c["forecast"],
        lstm_units=c["lstm_units"], num_layers=c["num_layers"],
        dropout=c["dropout"], dense_units=c["dense_units"],
        learning_rate=c["learning_rate"],
    )
    if verbose:
        model.summary()

    # 4. Train
    v = int(verbose)
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=c["patience"],
                      restore_best_weights=True, verbose=v),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=7,
                          min_lr=1e-6, verbose=v),
    ]
    spinner = Spinner() if not verbose else None
    if spinner:
        print("")
        spinner.start()

    history = model.fit(
        X_train, y_train,
        epochs=c["epochs"],
        batch_size=c["batch_size"],
        validation_split=0.1,
        callbacks=callbacks,
        verbose=v,
    )

    if spinner:
        spinner.stop()

    # 5. Predict
    train_pred_scaled = model.predict(X_train, verbose=0)
    test_pred_scaled  = model.predict(X_test,  verbose=0)

    train_actual = inverse_close(scaler, y_train[:, 0])
    train_pred   = inverse_close(scaler, train_pred_scaled[:, 0])
    test_actual  = inverse_close(scaler, y_test[:, 0])
    test_pred    = inverse_close(scaler, test_pred_scaled[:, 0])

    # 6. Evaluate
    evaluate(train_actual, train_pred, label="Train")
    evaluate(test_actual,  test_pred,  label="Test ")

    # 7. Next-day forecast
    next_prices = _forecast_next(model, scaler, df, c["features"], c["lookback"], c["forecast"])
    _print_forecast(ticker, df, next_prices)

    # 8. Save plot
    output_dir = Path("prediction_images")
    output_dir.mkdir(exist_ok=True)
    save_plot(ticker, train_dates, train_actual, test_dates, test_actual, test_pred,
              history, output_dir, lookback=c["lookback"])

    return model, scaler, history


# ──────────────────────────────────────────────
# BACKTEST
# ──────────────────────────────────────────────

def run_backtest(ticker: str, n_days: int,
                 lookback_override: int | None = None,
                 db_path: str = SQLITE_PATH,
                 verbose: bool = False) -> dict | None:
    ticker = ticker.strip().upper()
    c = _resolve_config(ticker,
                        lookback_override if lookback_override else LOOKBACK,
                        FORECAST, EPOCHS, None)

    df = load_data(ticker, db_path, features=c["features"])
    min_rows = c["lookback"] + n_days + 30
    if len(df) < min_rows:
        print(f"  [{ticker}] Data tidak cukup untuk backtest {n_days} hari "
              f"(ada {len(df)}, butuh minimal {min_rows}).")
        return None

    raw_all   = df[c["features"]].values.astype(np.float32)
    raw_train = raw_all[:-n_days]
    split_idx = int(len(raw_train) * c["train_split"])

    scaler       = MinMaxScaler()
    train_scaled = scaler.fit_transform(raw_train[:split_idx])

    X_train, y_train = make_sequences(train_scaled, c["lookback"], 1)
    if len(X_train) < 30:
        print(f"  [{ticker}] Sequence training tidak cukup.")
        return None

    model = build_model(
        c["lookback"], len(c["features"]), 1,
        lstm_units=c["lstm_units"], num_layers=c["num_layers"],
        dropout=c["dropout"], dense_units=c["dense_units"],
        learning_rate=c["learning_rate"],
    )

    spinner = Spinner(f"  Training {ticker} (backtest)") if not verbose else None
    if spinner:
        spinner.start()

    model.fit(
        X_train, y_train,
        epochs=c["epochs"],
        batch_size=c["batch_size"],
        validation_split=0.1,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=c["patience"],
                          restore_best_weights=True, verbose=int(verbose)),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=7,
                              min_lr=1e-6, verbose=int(verbose)),
        ],
        verbose=int(verbose),
    )

    if spinner:
        spinner.stop()

    rows = []
    for i in range(n_days, 0, -1):
        window = raw_all[-(i + c["lookback"]):-i]
        if len(window) < c["lookback"]:
            continue
        window_scaled = scaler.transform(window).reshape(1, c["lookback"], len(c["features"]))
        pred_scaled   = model(tf.constant(window_scaled.astype(np.float32)),
                              training=False).numpy()[0][0]
        pred_close   = inverse_close(scaler, [pred_scaled])[0]
        actual_close = df["Close"].iloc[-i]
        last_close   = df["Close"].iloc[-(i + 1)]
        pred_date    = df.index[-i].date()
        error_pct    = (pred_close - actual_close) / actual_close * 100
        correct_dir  = (pred_close >= last_close) == (actual_close >= last_close)
        rows.append({
            "Tanggal"   : pred_date,
            "Last Close": last_close,
            "Pred Close": pred_close,
            "Act Close" : actual_close,
            "Error%"    : error_pct,
            "Arah"      : "✓" if correct_dir else "✗",
        })

    if not rows:
        return None

    result_df = pd.DataFrame(rows)
    mape      = result_df["Error%"].abs().mean()
    dir_acc   = (result_df["Arah"] == "✓").mean() * 100
    n_benar   = int((result_df["Arah"] == "✓").sum())
    n         = len(rows)

    print(f"\n{'═'*82}")
    print(f"  Backtest LSTM — {ticker} — {n} hari terakhir  (lookback={c['lookback']})")
    print(f"  Catatan: model dilatih sekali pada data sebelum periode backtest")
    print(f"{'═'*82}")
    print(f"  {'Tanggal':<12} {'Last Close':>11} {'Pred Close':>11} "
          f"{'Act Close':>10} {'Error%':>7} {'Dir':>4}")
    print(f"  {'─'*12} {'─'*11} {'─'*11} {'─'*10} {'─'*7} {'─'*4}")
    for _, r in result_df.iterrows():
        print(f"  {str(r['Tanggal']):<12} {r['Last Close']:>11,.0f} "
              f"{r['Pred Close']:>11,.0f} {r['Act Close']:>10,.0f} "
              f"{r['Error%']:>+6.2f}% {r['Arah']:>4}")
    print(f"  {'─'*82}")
    print(f"  MAPE: {mape:.2f}%   |   Akurasi arah: {dir_acc:.1f}%   |   "
          f"Benar: {n_benar}/{n} hari")
    print(f"{'═'*82}\n")

    return {"ticker": ticker, "mape": round(mape, 2), "dir_acc": round(dir_acc, 1),
            "n_benar": n_benar, "n_total": n, "lookback": c["lookback"]}


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "BEI LSTM Stock Price Predictor\n"
            "================================\n"
            "Memprediksi harga penutupan hari berikutnya untuk saham BEI (IDX)\n"
            "menggunakan model stacked LSTM. Data dibaca dari bei_stocks.db\n"
            "(diisi oleh bei_stock_downloader.py).\n\n"
            "Konfigurasi optimal per ticker dimuat otomatis dari lstm_configs.json\n"
            "jika tersedia — tidak perlu set --lookback atau parameter lain secara manual.\n"
            "Untuk mencari konfigurasi optimal, gunakan lstm_config_search.py."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Parameter:
  --ticker      IDX ticker code (wajib). Contoh: BBCA, TLKM, ANTM.
  --lookback    Jumlah hari ke belakang per sequence. Nilai optimal berbeda per ticker —
                gunakan lstm_config_search.py untuk menemukannya. (default: 48)
  --forecast    Jumlah hari ke depan yang diprediksi sekaligus. (default: 1)
  --epochs      Batas maksimum epoch. Early stopping biasanya berhenti lebih awal. (default: 300)
  --db          Path ke SQLite DB. (default: bei_stocks.db)
  --train-split Proporsi data untuk training, sisanya untuk test. (default: 0.8)
  --train-all   Latih pada seluruh data; evaluasi test pada 20%% terakhir (overlap).
                Gunakan ini untuk prediksi final setelah tuning selesai.
  --save-config Simpan semua parameter run ini ke lstm_configs.json untuk ticker ini.
                Dipakai setelah menemukan konfigurasi optimal.
  --verbose     Tampilkan model summary, progress per epoch, dan pesan callback.
                Default: hanya hasil akhir yang muncul.

Output:
  - Tabel prediksi harga di terminal (last close + forecast +1d, +2d, dst.)
  - Plot disimpan ke prediction_images/{TICKER}_lstm_prediction.png

Alur kerja yang disarankan:
  1. Download data       : python bei_stock_downloader.py --ticker BBCA --years 5
  2. Cari lookback optimal: python lstm_config_search.py --ticker BBCA
  3. Prediksi             : python lstm_predictor.py --ticker BBCA
     (config optimal dimuat otomatis dari lstm_configs.json)

Contoh:
  python lstm_predictor.py --ticker BBCA
  python lstm_predictor.py --ticker TLKM --lookback 20 --epochs 150
  python lstm_predictor.py --ticker BBRI --forecast 3
  python lstm_predictor.py --ticker BBCA --train-all --save-config
  python lstm_predictor.py --ticker ANTM --lookback 28 --train-split 0.9 --verbose
  python lstm_predictor.py --ticker BBCA --backtest 30
        """
    )
    parser.add_argument("--ticker",      type=str, required=True,
                        help="IDX ticker code (wajib), contoh: BBCA")
    parser.add_argument("--lookback",    type=int, default=LOOKBACK,
                        help=f"Jumlah hari ke belakang per sequence. Diabaikan jika ada "
                             f"config di lstm_configs.json. (default: {LOOKBACK})")
    parser.add_argument("--forecast",    type=int, default=FORECAST,
                        help=f"Jumlah hari ke depan yang diprediksi sekaligus. (default: {FORECAST})")
    parser.add_argument("--epochs",      type=int, default=EPOCHS,
                        help=f"Batas maksimum epoch training; early stopping bisa berhenti lebih awal. "
                             f"(default: {EPOCHS})")
    parser.add_argument("--db",          type=str, default=SQLITE_PATH,
                        help=f"Path ke SQLite DB. (default: {SQLITE_PATH})")
    parser.add_argument("--train-split", type=float, default=None,
                        help="Proporsi data untuk training, contoh: 0.9 untuk 90/10 split. "
                             "Default: nilai dari lstm_configs.json atau 0.8.")
    parser.add_argument("--train-all",   action="store_true",
                        help="Latih pada seluruh data; evaluasi test pada 20%% data terakhir "
                             "(overlap dengan train). Prediksi lebih akurat untuk data terkini.")
    parser.add_argument("--save-config", action="store_true",
                        help="Simpan semua parameter run ini ke lstm_configs.json untuk ticker ini. "
                             "Gunakan setelah menemukan konfigurasi optimal.")
    parser.add_argument("--verbose",     action="store_true",
                        help="Tampilkan model summary, progress per epoch, dan pesan callback. "
                             "Default: hanya hasil akhir yang ditampilkan.")
    parser.add_argument("--backtest",    type=int, default=None, metavar="N",
                        help="Uji mundur N hari terakhir: bandingkan prediksi LSTM vs harga aktual. "
                             "Model dilatih sekali pada data sebelum periode backtest.")

    args = parser.parse_args()

    if args.backtest is not None:
        run_backtest(args.ticker, args.backtest,
                     lookback_override=args.lookback if args.lookback != LOOKBACK else None,
                     db_path=args.db, verbose=args.verbose)
    else:
        run(args.ticker, args.lookback, args.forecast, args.epochs, args.db,
            verbose=args.verbose, train_all=args.train_all,
            train_split_override=args.train_split)

    if args.save_config:
        from datetime import date
        c = _resolve_config(args.ticker, args.lookback, args.forecast,
                            args.epochs, args.train_split)
        save_ticker_config(args.ticker, {**c, "last_updated": str(date.today())})
