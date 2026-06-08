"""
BEI Linear Regression Config Search
=====================================
Mencari kombinasi lookback dan alpha (Ridge) optimal per ticker
menggunakan grid search + TimeSeriesSplit cross-validation.

Metrik evaluasi:
  - MAE (Mean Absolute Error) DayReturn_Pct — lebih kecil lebih baik
  - Direction Accuracy — % prediksi naik/turun yang benar

Hasil terbaik disimpan ke ridge_configs.json.

Usage:
    python ridge_config_search.py --ticker DMAS
    python ridge_config_search.py --tickers DMAS BBCA ANTM
    python ridge_config_search.py --ticker DMAS --lookbacks 1 3 5 7 10 14 20
"""

import argparse
import json
import sqlite3
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

SQLITE_PATH = "bei_stocks.db"
LR_CONFIGS  = "ridge_configs.json"

DEFAULT_LOOKBACKS = [1, 2, 3, 5, 7, 10, 14, 20]
DEFAULT_ALPHAS    = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
N_SPLITS          = 5


def load_data(ticker: str) -> pd.DataFrame:
    conn = sqlite3.connect(SQLITE_PATH)
    df = pd.read_sql_query(
        "SELECT Date, Volume, DayReturn_Pct FROM daily_prices "
        "WHERE Ticker = ? ORDER BY Date ASC",
        conn, params=(ticker,)
    )
    conn.close()
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    df.dropna(subset=["DayReturn_Pct", "Volume"], inplace=True)
    return df


def build_features(df: pd.DataFrame, lookback: int) -> tuple[pd.DataFrame, pd.Series]:
    feat = {}
    for lag in range(1, lookback + 1):
        feat[f"return_lag{lag}"] = df["DayReturn_Pct"].shift(lag)
        feat[f"volume_lag{lag}"] = df["Volume"].shift(lag)
    feat_df = pd.DataFrame(feat, index=df.index)
    feat_df["target"] = df["DayReturn_Pct"]
    feat_df.dropna(inplace=True)
    X = feat_df.drop(columns=["target"])
    y = feat_df["target"]
    return X, y


def cv_score(X: pd.DataFrame, y: pd.Series, alpha: float) -> tuple[float, float]:
    tscv    = TimeSeriesSplit(n_splits=N_SPLITS)
    vol_cols = [c for c in X.columns if "volume" in c]
    maes, dir_accs = [], []

    for train_idx, val_idx in tscv.split(X):
        X_tr, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
        y_tr, y_val = y.iloc[train_idx],        y.iloc[val_idx]

        scaler = StandardScaler()
        X_tr[vol_cols]  = scaler.fit_transform(X_tr[vol_cols])
        X_val[vol_cols] = scaler.transform(X_val[vol_cols])

        model = Ridge(alpha=alpha)
        model.fit(X_tr, y_tr)
        pred = model.predict(X_val)

        maes.append(mean_absolute_error(y_val, pred))
        dir_accs.append(float(np.mean(np.sign(pred) == np.sign(y_val.values))))

    return float(np.mean(maes)), float(np.mean(dir_accs) * 100)


def search_ticker(ticker: str, lookbacks: list[int], alphas: list[float]) -> dict:
    ticker = ticker.strip().upper()
    print(f"\n  Ticker: {ticker}")
    print(f"  {'Lookback':>8}  {'Alpha':>8}  {'MAE':>8}  {'DirAcc':>8}")
    print(f"  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}")

    df = load_data(ticker)
    if len(df) < 30:
        print(f"  SKIP: data tidak cukup ({len(df)} baris)")
        return {}

    best_mae     = float("inf")
    best_result  = {}

    for lookback in lookbacks:
        X, y = build_features(df, lookback)
        if len(X) < N_SPLITS * 10:
            continue
        for alpha in alphas:
            mae, dir_acc = cv_score(X, y, alpha)
            marker = " ◀ best" if mae < best_mae else ""
            print(f"  {lookback:>8}  {alpha:>8}  {mae:>8.4f}  {dir_acc:>7.1f}%{marker}")
            if mae < best_mae:
                best_mae    = mae
                best_result = {
                    "lookback"    : lookback,
                    "alpha"       : alpha,
                    "cv_mae"      : round(mae, 4),
                    "cv_dir_acc"  : round(dir_acc, 2),
                    "data_rows"   : len(df),
                    "last_updated": str(date.today()),
                }

    if best_result:
        print(f"\n  Best → lookback={best_result['lookback']}, alpha={best_result['alpha']}, "
              f"MAE={best_result['cv_mae']:.4f}%, DirAcc={best_result['cv_dir_acc']:.1f}%")
    else:
        print(f"\n  SKIP: tidak ada kombinasi dengan data cukup")
    return best_result


def save_configs(updates: dict[str, dict]) -> None:
    path = Path(LR_CONFIGS)
    configs = json.loads(path.read_text()) if path.exists() else {}
    configs.update(updates)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(configs, indent=2))
    tmp.replace(path)
    print(f"\n  Config disimpan → {LR_CONFIGS}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="ridge_config_search.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "BEI Linear Regression Config Search\n"
            "=====================================\n"
            "Mencari lookback dan alpha Ridge optimal per ticker via grid search\n"
            "dengan TimeSeriesSplit cross-validation (5 fold).\n"
            "Hasil disimpan ke ridge_configs.json dan langsung dipakai oleh ridge_predictor.py."
        ),
        epilog=(
            "Contoh:\n"
            "  python ridge_config_search.py --ticker DMAS\n"
            "  python ridge_config_search.py --tickers DMAS BBCA ANTM\n"
            "  python ridge_config_search.py --ticker DMAS --lookbacks 3 5 7 10 --alphas 0.1 1 10\n"
        ),
    )
    parser.add_argument("--ticker",    help="Satu ticker IDX")
    parser.add_argument("--tickers",   nargs="+", help="Beberapa ticker sekaligus")
    parser.add_argument("--lookbacks", nargs="+", type=int, default=DEFAULT_LOOKBACKS,
                        metavar="N", help=f"Nilai lookback yang dicoba (default: {DEFAULT_LOOKBACKS})")
    parser.add_argument("--alphas",    nargs="+", type=float, default=DEFAULT_ALPHAS,
                        metavar="A", help=f"Nilai alpha Ridge yang dicoba (default: {DEFAULT_ALPHAS})")
    args = parser.parse_args()

    tickers = args.tickers or ([args.ticker] if args.ticker else [])
    if not tickers:
        parser.error("Tentukan --ticker atau --tickers")

    print(f"\n{'═'*55}")
    print(f"  LR Config Search")
    print(f"  Lookbacks : {args.lookbacks}")
    print(f"  Alphas    : {args.alphas}")
    print(f"  CV splits : {N_SPLITS}")
    print(f"{'═'*55}")

    results = {}
    for t in tickers:
        res = search_ticker(t, args.lookbacks, args.alphas)
        if res:
            results[t.strip().upper()] = res

    if results:
        save_configs(results)
