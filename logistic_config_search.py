"""
BEI Logistic Regression Classifier — Config Search
=====================================================
Mencari konfigurasi optimal (lookback + C) per ticker menggunakan
LogisticRegression untuk memprediksi ARAH (naik/turun) hari berikutnya.

Fitur yang digunakan:
  - ret_lag1..N     : DayReturn_Pct N hari terakhir
  - volchg_lag1..N  : % perubahan volume N hari terakhir
  - momentum_3      : jumlah return 3 hari terakhir (trend jangka pendek)
  - jkse_lag1       : return JKSE kemarin (konteks pasar)
  - abs_ret_lag1    : besar return kemarin (sinyal volatilitas)

Target: 1 = besok naik (DayReturn_Pct > 0), 0 = turun/flat

Hasil disimpan ke logistic_configs.json.

Usage:
    python logistic_config_search.py                           # semua ticker di watchlist.txt
    python logistic_config_search.py --tickers DMAS BBCA
    python logistic_config_search.py --lookbacks 1 3 5 7
"""

import argparse
import sqlite3
from datetime import date

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from logistic_classifier import build_features, load_jkse
from utils import load_watchlist, save_configs as _save_configs

SQLITE_PATH    = "bei_stocks.db"
LR_CLS_CONFIGS = "logistic_configs.json"

DEFAULT_LOOKBACKS = [1, 2, 3, 5, 7, 10, 14]
DEFAULT_C_VALUES  = [0.01, 0.1, 1.0, 10.0, 100.0]
N_SPLITS          = 5


def load_all_tickers() -> list[str]:
    conn = sqlite3.connect(SQLITE_PATH)
    df   = pd.read_sql_query("SELECT DISTINCT Ticker FROM daily_prices ORDER BY Ticker", conn)
    conn.close()
    return df["Ticker"].tolist()


def load_data(ticker: str) -> pd.DataFrame:
    conn = sqlite3.connect(SQLITE_PATH)
    df   = pd.read_sql_query(
        "SELECT Date, Close, Volume, DayReturn_Pct FROM daily_prices "
        "WHERE Ticker = ? ORDER BY Date ASC",
        conn, params=(ticker,)
    )
    conn.close()
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    return df


def cv_score(X: pd.DataFrame, y: pd.Series, c: float) -> float:
    tscv    = TimeSeriesSplit(n_splits=N_SPLITS)
    accs    = []
    for train_idx, val_idx in tscv.split(X):
        X_tr, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
        y_tr, y_val = y.iloc[train_idx],        y.iloc[val_idx]

        scaler = StandardScaler()
        X_tr   = pd.DataFrame(scaler.fit_transform(X_tr),  columns=X_tr.columns)
        X_val  = pd.DataFrame(scaler.transform(X_val),     columns=X_val.columns)

        model = LogisticRegression(C=c, class_weight="balanced",
                                   max_iter=1000, random_state=42)
        model.fit(X_tr, y_tr)
        accs.append(float(np.mean(model.predict(X_val) == y_val.values)))

    return float(np.mean(accs) * 100)


def search_ticker(ticker: str, jkse: pd.Series,
                  lookbacks: list[int], c_values: list[float]) -> dict:
    ticker = ticker.strip().upper()
    print(f"\n  Ticker: {ticker}")
    print(f"  {'Lookback':>8}  {'C':>8}  {'DirAcc':>8}")
    print(f"  {'─'*8}  {'─'*8}  {'─'*8}")

    df = load_data(ticker)
    df = df[df["DayReturn_Pct"].notna() & (df["Volume"] > 0)]

    if len(df) < 50:
        print(f"  SKIP: data tidak cukup ({len(df)} baris)")
        return {}

    best_acc    = 0.0
    best_result = {}

    for lookback in lookbacks:
        X, y = build_features(df, jkse, lookback)
        if len(X) < N_SPLITS * 10:
            continue
        for c in c_values:
            acc    = cv_score(X, y, c)
            marker = " ◀ best" if acc > best_acc else ""
            print(f"  {lookback:>8}  {c:>8}  {acc:>7.1f}%{marker}")
            if acc > best_acc:
                best_acc    = acc
                best_result = {
                    "lookback"    : lookback,
                    "C"           : c,
                    "cv_dir_acc"  : round(acc, 2),
                    "data_rows"   : len(df),
                    "last_updated": str(date.today()),
                }

    if best_result:
        print(f"\n  Best → lookback={best_result['lookback']}, C={best_result['C']}, "
              f"DirAcc={best_result['cv_dir_acc']:.1f}%")
    return best_result


def save_configs(updates: dict[str, dict]) -> None:
    _save_configs(LR_CLS_CONFIGS, updates)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="logistic_config_search.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "BEI Logistic Regression Classifier — Config Search\n"
            "=====================================================\n"
            "Mencari lookback dan C optimal per ticker menggunakan LogisticRegression\n"
            "dengan TimeSeriesSplit cross-validation (5 fold).\n"
            "Hasil disimpan ke logistic_configs.json."
        ),
        epilog=(
            "Alur kerja yang disarankan:\n"
            "  1. Riset config semua watchlist : python logistic_config_search.py\n"
            "  2. Prediksi arah                : python logistic_classifier.py --all\n"
            "  3. Backtest                     : python logistic_classifier.py --all --backtest 30\n"
            "\n"
            "Contoh:\n"
            "  python logistic_config_search.py                          # semua dari watchlist.txt\n"
            "  python logistic_config_search.py --tickers DMAS BBCA\n"
            "  python logistic_config_search.py --lookbacks 1 3 5 7 --c-values 0.1 1 10\n"
        ),
    )
    parser.add_argument("--tickers",   nargs="+", help="Ticker spesifik (default: semua dari watchlist.txt)")
    parser.add_argument("--lookbacks", nargs="+", type=int,   default=DEFAULT_LOOKBACKS, metavar="N")
    parser.add_argument("--c-values",  nargs="+", type=float, default=DEFAULT_C_VALUES,  metavar="C")
    args = parser.parse_args()

    tickers = args.tickers or load_watchlist()
    jkse    = load_jkse()

    print(f"\n{'═'*55}")
    print(f"  LR Classifier Config Search")
    print(f"  Tickers   : {tickers}")
    print(f"  Lookbacks : {args.lookbacks}")
    print(f"  C values  : {args.c_values}")
    print(f"  CV splits : {N_SPLITS}")
    print(f"  Fitur     : ret_lag, volchg_lag, momentum_3/5, abs_ret, jkse")
    print(f"{'═'*55}")

    results = {}
    for t in tickers:
        res = search_ticker(t, jkse, args.lookbacks, args.c_values)
        if res:
            results[t.strip().upper()] = res

    if results:
        save_configs(results)

        print(f"\n{'═'*50}")
        print(f"  Ringkasan Hasil")
        print(f"{'═'*50}")
        print(f"  {'Ticker':<8}  {'Lookback':>8}  {'C':>8}  {'DirAcc':>8}")
        print(f"  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}")
        for t, r in sorted(results.items(), key=lambda x: -x[1]["cv_dir_acc"]):
            print(f"  {t:<8}  {r['lookback']:>8}  {r['C']:>8}  {r['cv_dir_acc']:>7.1f}%")
        print(f"{'═'*50}\n")
