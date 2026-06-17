"""
Lookback Hyperparameter Search
================================
Trains an LSTM model for every lookback value in [start..end] and records
Test MAE, RMSE, MAPE. Best config per ticker disimpan otomatis ke lstm_configs.json.

Output:
    ticker_configs_research/{TICKER}_lookback_search.csv  — metrik per lookback
    ticker_configs_research/{TICKER}_lookback_search.png  — plot kurva MAPE/MAE/RMSE

Usage:
    python lstm_config_search.py --ticker BBCA
    python lstm_config_search.py --ticker BBCA --start 3 --end 60
"""

import argparse
import os
import sqlite3
import time
import warnings
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential

RESEARCH_DIR = Path("ticker_configs_research")
RESEARCH_DIR.mkdir(exist_ok=True)

# ── fixed config ──────────────────────────────────────────────────────────────
SQLITE_PATH  = "bei_stocks.db"
FEATURES     = ["Close", "Volume", "GainLoss_Pct", "DayReturn_Pct"]
TRAIN_SPLIT  = 0.8

LSTM_UNITS   = 64
NUM_LAYERS   = 2
DROPOUT      = 0.2
DENSE_UNITS  = 32

EPOCHS       = 80       # cap kept low for speed; early stopping handles convergence
BATCH_SIZE   = 32
LR           = 0.001
PATIENCE     = 12

SEED         = 42       # fixed seed → reproducible results across runs


# ── helpers ───────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_data(ticker: str) -> pd.DataFrame:
    conn = sqlite3.connect(SQLITE_PATH)
    df = pd.read_sql_query(
        "SELECT Date, Close, Volume, GainLoss_Pct, DayReturn_Pct "
        "FROM daily_prices WHERE Ticker = ? ORDER BY Date ASC",
        conn, params=(ticker.upper().strip(),)
    )
    conn.close()
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    df.dropna(subset=FEATURES, inplace=True)
    return df


def make_sequences(data: np.ndarray, lookback: int) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i : i + lookback])
        y.append(data[i + lookback, 0])          # col 0 = Close
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def preprocess(df: pd.DataFrame, lookback: int) -> tuple:
    raw      = df[FEATURES].values.astype(np.float32)
    split    = int(len(raw) * TRAIN_SPLIT)

    scaler        = MinMaxScaler()
    train_scaled  = scaler.fit_transform(raw[:split])
    test_full     = scaler.transform(np.concatenate([raw[split - lookback:split], raw[split:]], axis=0))

    X_train, y_train = make_sequences(train_scaled, lookback)
    X_test,  y_test  = make_sequences(test_full,    lookback)
    return X_train, y_train, X_test, y_test, scaler


def inverse_close(scaler, vals) -> np.ndarray:
    n = scaler.scale_.shape[0]
    dummy = np.zeros((len(vals), n), dtype=np.float32)
    dummy[:, 0] = vals
    return scaler.inverse_transform(dummy)[:, 0]


def build_model(lookback: int, n_features: int) -> Sequential:
    model = Sequential()
    for i in range(NUM_LAYERS):
        rs = (i < NUM_LAYERS - 1)
        kwargs = dict(return_sequences=rs)
        if i == 0:
            kwargs["input_shape"] = (lookback, n_features)
        model.add(LSTM(LSTM_UNITS, **kwargs))
        model.add(Dropout(DROPOUT))
    model.add(Dense(DENSE_UNITS, activation="relu"))
    model.add(Dense(1))
    model.compile(optimizer=tf.keras.optimizers.Adam(LR), loss="mse")
    return model


def train_and_eval(df: pd.DataFrame, lookback: int) -> dict:
    set_seed(SEED)
    X_train, y_train, X_test, y_test, scaler = preprocess(df, lookback)

    if len(X_train) < 30:
        return None                             # not enough data for this lookback

    model = build_model(lookback, len(FEATURES))
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=PATIENCE,
                      restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6,
                          min_lr=1e-6, verbose=0),
    ]
    model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=0,
    )

    pred_scaled  = model(X_test, training=False).numpy().flatten()
    actual       = inverse_close(scaler, y_test)
    predicted    = inverse_close(scaler, pred_scaled)

    mae  = mean_absolute_error(actual, predicted)
    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    mape = float(np.mean(np.abs((actual - predicted) / actual)) * 100)

    # how many epochs actually ran (best epoch = total - patience)
    stopped_epoch = len(model.history.history["loss"])

    tf.keras.backend.clear_session()            # release GPU/CPU memory between runs
    return {
        "lookback"     : lookback,
        "train_seq"    : len(X_train),
        "test_seq"     : len(X_test),
        "epochs_run"   : stopped_epoch,
        "mae_idr"      : round(mae,  2),
        "rmse_idr"     : round(rmse, 2),
        "mape_pct"     : round(mape, 4),
    }


# ── plot ──────────────────────────────────────────────────────────────────────

def save_plot(ticker: str, results_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    fig.suptitle(f"{ticker} — Lookback Hyperparameter Search  (LSTM, seed={SEED})",
                 fontsize=13)

    best_mape = results_df.loc[results_df["mape_pct"].idxmin()]

    for ax, col, label, color in [
        (axes[0], "mape_pct", "MAPE (%)",       "royalblue"),
        (axes[1], "mae_idr",  "MAE (IDR)",       "darkorange"),
        (axes[2], "rmse_idr", "RMSE (IDR)",      "seagreen"),
    ]:
        ax.plot(results_df["lookback"], results_df[col], marker="o",
                markersize=4, linewidth=1.5, color=color)
        ax.set_ylabel(label, fontsize=10)
        ax.grid(alpha=0.3)

    # mark best MAPE on all three subplots
    for ax, col in zip(axes, ["mape_pct", "mae_idr", "rmse_idr"]):
        val = results_df.loc[best_mape.name, col]
        ax.axvline(best_mape["lookback"], color="red", linestyle="--", linewidth=1, alpha=0.7)
        ax.annotate(f'best={int(best_mape["lookback"])}',
                    xy=(best_mape["lookback"], val),
                    xytext=(8, 6), textcoords="offset points",
                    fontsize=8, color="red")

    axes[-1].set_xlabel("Lookback (days)", fontsize=10)
    plt.tight_layout()
    path = str(RESEARCH_DIR / f"{ticker}_lookback_search.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Plot saved → {path}")


# ── main ──────────────────────────────────────────────────────────────────────

def run(ticker: str, start: int, end: int) -> pd.DataFrame | None:
    print(f"\n{'═'*60}")
    print(f"  Lookback Search: {ticker}  |  range {start}–{end}  |  {end-start+1} models")
    print(f"  Fixed seed={SEED}  epochs≤{EPOCHS}  patience={PATIENCE}")
    print(f"{'═'*60}\n")

    df = load_data(ticker)
    print(f"  Data: {len(df)} rows  ({df.index[0].date()} → {df.index[-1].date()})\n")

    results = []
    total   = end - start + 1

    for i, lb in enumerate(range(start, end + 1), 1):
        t0 = time.time()
        row = train_and_eval(df, lb)
        elapsed = time.time() - t0

        if row is None:
            print(f"  [{i:2d}/{total}]  lookback={lb:2d}  SKIP (not enough data)")
            continue

        results.append(row)
        print(
            f"  [{i:2d}/{total}]  lookback={lb:2d}  "
            f"MAPE={row['mape_pct']:6.2f}%  "
            f"MAE={row['mae_idr']:7,.0f}  "
            f"RMSE={row['rmse_idr']:7,.0f}  "
            f"epochs={row['epochs_run']:3d}  "
            f"({elapsed:.1f}s)"
        )

    if not results:
        print("No results collected.")
        return

    df_res = pd.DataFrame(results)

    # ── summary ──────────────────────────────────────────────────────────────
    best_mape = df_res.loc[df_res["mape_pct"].idxmin()]
    best_mae  = df_res.loc[df_res["mae_idr"].idxmin()]
    best_rmse = df_res.loc[df_res["rmse_idr"].idxmin()]

    print(f"\n{'═'*60}")
    print(f"  RESULTS SUMMARY — {ticker}")
    print(f"{'─'*60}")
    print(f"  Best MAPE : lookback={int(best_mape['lookback']):2d}  →  {best_mape['mape_pct']:.4f}%")
    print(f"  Best MAE  : lookback={int(best_mae['lookback']):2d}  →  {best_mae['mae_idr']:,.0f} IDR")
    print(f"  Best RMSE : lookback={int(best_rmse['lookback']):2d}  →  {best_rmse['rmse_idr']:,.0f} IDR")
    print(f"{'─'*60}")

    # top 5 by MAPE
    print(f"\n  Top 5 lookback values by MAPE:")
    top5 = df_res.nsmallest(5, "mape_pct")[
        ["lookback", "mape_pct", "mae_idr", "rmse_idr", "train_seq", "epochs_run"]
    ].reset_index(drop=True)
    top5.index += 1
    print(top5.to_string())

    print(f"{'═'*60}\n")

    # ── save CSV ──────────────────────────────────────────────────────────────
    csv_path = str(RESEARCH_DIR / f"{ticker}_lookback_search.csv")
    df_res.to_csv(csv_path, index=False)
    print(f"  Full results saved → {csv_path}")

    # ── plot ──────────────────────────────────────────────────────────────────
    save_plot(ticker, df_res)

    return df_res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="lstm_config_search.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Lookback Hyperparameter Search — LSTM\n"
            "=======================================\n"
            "Melatih LSTM untuk setiap nilai lookback dalam rentang [start..end]\n"
            "dan mencatat Test MAE, RMSE, MAPE. Best config disimpan otomatis\n"
            "ke lstm_configs.json. Output CSV + PNG disimpan ke ticker_configs_research/."
        ),
        epilog=(
            "Alur kerja yang disarankan:\n"
            "  1. Download data       : python bei_stock_downloader.py --ticker BBCA --years 5\n"
            "  2. Cari lookback optimal: python lstm_config_search.py --ticker BBCA\n"
            "  3. Jalankan prediksi   : python lstm_predictor.py --ticker BBCA\n"
            "\n"
            "Contoh:\n"
            "  python lstm_config_search.py --ticker BBCA\n"
            "  python lstm_config_search.py --ticker BBCA --start 3 --end 60\n"
        ),
    )
    parser.add_argument("--ticker", type=str, required=True, help="Kode saham IDX, contoh: BBCA")
    parser.add_argument("--start",  type=int, default=3,  help="Lookback start (default: 3)")
    parser.add_argument("--end",    type=int, default=60, help="Lookback end   (default: 60)")
    args = parser.parse_args()
    run(args.ticker, args.start, args.end)
