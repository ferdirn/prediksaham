"""
Batch Configuration Search
===========================
Runs lookback hyperparameter search (3–60) for every ticker in watchlist.txt,
saves the best config per ticker into lstm_configs.json, and prints a final
summary table.

Skips tickers that already have a saved config (use --force to re-run them).

Usage:
    python lstm_batch_config_search.py
    python lstm_batch_config_search.py --force          # re-run all, including saved ones
    python lstm_batch_config_search.py --tickers ANTM TLKM
"""

import argparse
import json
import os
import sqlite3
import time
import warnings
from datetime import date
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

# ── fixed search config ───────────────────────────────────────────────────────
SQLITE_PATH    = "bei_stocks.db"
WATCHLIST_FILE = "watchlist.txt"
TICKER_CONFIGS = "lstm_configs.json"
RESEARCH_DIR   = Path("ticker_configs_research")
RESEARCH_DIR.mkdir(exist_ok=True)

LOOKBACK_START = 3
LOOKBACK_END   = 60

FEATURES       = ["Close", "Volume", "GainLoss_Pct", "DayReturn_Pct"]
TRAIN_SPLIT    = 0.8

LSTM_UNITS     = 64
NUM_LAYERS     = 2
DROPOUT        = 0.2
DENSE_UNITS    = 32

EPOCHS         = 80
BATCH_SIZE     = 32
LR             = 0.001
PATIENCE       = 12

SEED           = 42


# ── helpers ───────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    np.random.seed(seed)
    tf.random.set_seed(seed)


def read_watchlist() -> list:
    lines = Path(WATCHLIST_FILE).read_text().splitlines()
    return [l.strip() for l in lines if l.strip() and not l.startswith("#")]


def load_configs() -> dict:
    path = Path(TICKER_CONFIGS)
    return json.loads(path.read_text()) if path.exists() else {}


def save_configs(configs: dict):
    Path(TICKER_CONFIGS).write_text(json.dumps(configs, indent=2))


def load_ticker_data(ticker: str) -> pd.DataFrame:
    conn = sqlite3.connect(SQLITE_PATH)
    df = pd.read_sql_query(
        "SELECT Date, Close, Open, High, Low, Volume, GainLoss_Pct, DayReturn_Pct "
        "FROM daily_prices WHERE Ticker = ? ORDER BY Date ASC",
        conn, params=(ticker.upper(),)
    )
    conn.close()
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    df.dropna(subset=FEATURES, inplace=True)
    return df


def make_sequences(data: np.ndarray, lookback: int):
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i : i + lookback])
        y.append(data[i + lookback, 0])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def preprocess(df: pd.DataFrame, lookback: int):
    raw   = df[FEATURES].values.astype(np.float32)
    split = int(len(raw) * TRAIN_SPLIT)

    scaler       = MinMaxScaler()
    train_scaled = scaler.fit_transform(raw[:split])
    test_full    = scaler.transform(np.concatenate([raw[split - lookback:split], raw[split:]], axis=0))

    X_train, y_train = make_sequences(train_scaled, lookback)
    X_test,  y_test  = make_sequences(test_full,    lookback)
    return X_train, y_train, X_test, y_test, scaler


def inverse_close(scaler, vals):
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


def train_and_eval(df: pd.DataFrame, lookback: int) -> dict | None:
    set_seed(SEED)
    X_train, y_train, X_test, y_test, scaler = preprocess(df, lookback)
    if len(X_train) < 30:
        return None

    model = build_model(lookback, len(FEATURES))
    model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=PATIENCE,
                          restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6,
                              min_lr=1e-6, verbose=0),
        ],
        verbose=0,
    )

    pred_scaled = model(X_test, training=False).numpy().flatten()
    actual      = inverse_close(scaler, y_test)
    predicted   = inverse_close(scaler, pred_scaled)

    mape = float(np.mean(np.abs((actual - predicted) / actual)) * 100)
    mae  = float(mean_absolute_error(actual, predicted))
    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))

    tf.keras.backend.clear_session()
    return {
        "lookback"  : lookback,
        "mape_pct"  : round(mape, 4),
        "mae_idr"   : round(mae,  2),
        "rmse_idr"  : round(rmse, 2),
        "train_seq" : len(X_train),
        "test_seq"  : len(X_test),
    }


def save_plot(ticker: str, df_res: pd.DataFrame):
    best = df_res.loc[df_res["mape_pct"].idxmin()]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df_res["lookback"], df_res["mape_pct"], marker="o", markersize=3,
            linewidth=1.5, color="royalblue")
    ax.axvline(best["lookback"], color="red", linestyle="--", linewidth=1, alpha=0.7)
    ax.annotate(f'best={int(best["lookback"])}  MAPE={best["mape_pct"]:.2f}%',
                xy=(best["lookback"], best["mape_pct"]),
                xytext=(8, 6), textcoords="offset points", fontsize=9, color="red")
    ax.set_title(f"{ticker} — Lookback Search (seed={SEED})", fontsize=12)
    ax.set_xlabel("Lookback (days)")
    ax.set_ylabel("Test MAPE (%)")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(RESEARCH_DIR / f"{ticker}_lookback_search.png"), dpi=130)
    plt.close()


# ── per-ticker search ─────────────────────────────────────────────────────────

def search_ticker(ticker: str) -> dict | None:
    ticker = ticker.upper()
    print(f"\n{'═'*60}")
    print(f"  Searching: {ticker}  (lookback {LOOKBACK_START}–{LOOKBACK_END})")
    print(f"{'═'*60}")

    try:
        df = load_ticker_data(ticker)
    except Exception as e:
        print(f"  ERROR loading data: {e}")
        return None

    print(f"  Data: {len(df)} rows  ({df.index[0].date()} → {df.index[-1].date()})")
    if len(df) < 100:
        print(f"  SKIP: not enough data ({len(df)} rows < 100)")
        return None

    results = []
    total = LOOKBACK_END - LOOKBACK_START + 1
    t_start = time.time()

    for i, lb in enumerate(range(LOOKBACK_START, LOOKBACK_END + 1), 1):
        t0  = time.time()
        row = train_and_eval(df, lb)
        if row is None:
            continue
        results.append(row)
        elapsed = time.time() - t0
        print(
            f"  [{i:2d}/{total}]  lb={lb:2d}  "
            f"MAPE={row['mape_pct']:5.2f}%  "
            f"MAE={row['mae_idr']:6,.0f}  ({elapsed:.1f}s)"
        )

    if not results:
        print("  No results.")
        return None

    df_res  = pd.DataFrame(results)
    best    = df_res.loc[df_res["mape_pct"].idxmin()]

    elapsed_total = time.time() - t_start
    print(f"\n  ✅ Best lookback={int(best['lookback'])}  MAPE={best['mape_pct']:.4f}%"
          f"  MAE={best['mae_idr']:,.0f} IDR  ({elapsed_total/60:.1f} min)")

    # Save search CSV and plot
    df_res.to_csv(str(RESEARCH_DIR / f"{ticker}_lookback_search.csv"), index=False)
    save_plot(ticker, df_res)

    return {
        "lookback"     : int(best["lookback"]),
        "forecast"     : 1,
        "epochs"       : 300,
        "patience"     : 25,
        "seed"         : SEED,
        "features"     : FEATURES,
        "lstm_units"   : LSTM_UNITS,
        "num_layers"   : NUM_LAYERS,
        "dropout"      : DROPOUT,
        "dense_units"  : DENSE_UNITS,
        "batch_size"   : BATCH_SIZE,
        "learning_rate": LR,
        "train_split"  : TRAIN_SPLIT,
        "search_mape"  : round(float(best["mape_pct"]), 4),
        "search_mae"   : round(float(best["mae_idr"]),  2),
        "search_rmse"  : round(float(best["rmse_idr"]), 2),
        "data_rows"    : len(df),
        "notes"        : f"Best from lookback search {LOOKBACK_START}–{LOOKBACK_END}, seed={SEED}",
        "last_updated" : str(date.today()),
    }


# ── main ──────────────────────────────────────────────────────────────────────

def run(tickers: list, force: bool):
    configs  = load_configs()
    summary  = []
    t_total  = time.time()

    for ticker in tickers:
        ticker = ticker.upper()
        if not force and ticker in configs:
            print(f"\n  SKIP {ticker} — config already saved (use --force to re-run)")
            cfg = configs[ticker]
            summary.append({
                "ticker"  : ticker,
                "lookback": cfg.get("lookback", "—"),
                "mape"    : cfg.get("search_mape", cfg.get("mape", "—")),
                "mae"     : cfg.get("search_mae",  cfg.get("mae",  "—")),
                "status"  : "skipped (cached)",
            })
            continue

        cfg = search_ticker(ticker)
        if cfg is None:
            summary.append({"ticker": ticker, "lookback": "—", "mape": "—", "mae": "—",
                             "status": "failed / no data"})
            continue

        configs[ticker] = cfg
        save_configs(configs)
        summary.append({
            "ticker"  : ticker,
            "lookback": cfg["lookback"],
            "mape"    : f"{cfg['search_mape']:.2f}%",
            "mae"     : f"{cfg['search_mae']:,.0f} IDR",
            "status"  : "done",
        })

    # ── final report ─────────────────────────────────────────────────────────
    elapsed = (time.time() - t_total) / 60
    print(f"\n\n{'═'*60}")
    print(f"  BATCH SEARCH COMPLETE — {elapsed:.1f} min total")
    print(f"{'═'*60}")
    df_summary = pd.DataFrame(summary)
    print(df_summary.to_string(index=False))
    print(f"\n  Configs saved → {TICKER_CONFIGS}")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch lookback search for all watchlist tickers")
    parser.add_argument("--tickers", nargs="+", default=None,
                        help="Subset of tickers to search (default: all in watchlist.txt)")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if config already saved")
    args = parser.parse_args()

    tickers = args.tickers if args.tickers else read_watchlist()
    print(f"\n  Tickers to process: {tickers}")
    run(tickers, args.force)
