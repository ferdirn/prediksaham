"""
BEI Linear Regression Predictor
=================================
Memprediksi DayReturn_Pct hari berikutnya menggunakan regresi linear Ridge
dengan fitur lag DayReturn_Pct dan Volume.

Config optimal per ticker dimuat otomatis dari ridge_configs.json jika tersedia.
Untuk mencari config optimal, gunakan ridge_config_search.py.

Usage:
    python ridge_predictor.py --ticker DMAS
    python ridge_predictor.py --all
    python ridge_predictor.py --ticker ANTM --lookback 7 --alpha 0.1
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
from sklearn.preprocessing import StandardScaler

SQLITE_PATH  = "bei_stocks.db"
LR_CONFIGS   = "ridge_configs.json"
WATCHLIST    = "watchlist.txt"

DEFAULT_CONFIG = {
    "lookback": 5,
    "alpha"   : 1.0,
}


def load_config(ticker: str) -> dict:
    path = Path(LR_CONFIGS)
    if path.exists():
        configs = json.loads(path.read_text())
        if ticker in configs:
            return configs[ticker]
    return DEFAULT_CONFIG.copy()


def load_watchlist() -> list[str]:
    lines = Path(WATCHLIST).read_text().splitlines()
    return [l.strip() for l in lines if l.strip() and not l.startswith("#")]


def load_data(ticker: str) -> pd.DataFrame:
    conn = sqlite3.connect(SQLITE_PATH)
    df = pd.read_sql_query(
        "SELECT Date, Close, Volume, DayReturn_Pct FROM daily_prices "
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


def build_next_features(df: pd.DataFrame, lookback: int) -> pd.DataFrame:
    row = {}
    returns = df["DayReturn_Pct"].values
    volumes = df["Volume"].values
    for lag in range(1, lookback + 1):
        row[f"return_lag{lag}"] = returns[-lag]
        row[f"volume_lag{lag}"] = volumes[-lag]
    return pd.DataFrame([row])


def predict_ticker(ticker: str, lookback_override: int | None = None,
                   alpha_override: float | None = None) -> dict | None:
    ticker = ticker.strip().upper()
    cfg    = load_config(ticker)

    lookback = lookback_override if lookback_override is not None else cfg["lookback"]
    alpha    = alpha_override    if alpha_override    is not None else cfg["alpha"]

    df = load_data(ticker)
    if df.empty:
        return None

    X, y = build_features(df, lookback)

    vol_cols = [c for c in X.columns if "volume" in c]
    scaler   = StandardScaler()
    X        = X.copy()
    X[vol_cols] = scaler.fit_transform(X[vol_cols])

    split           = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)

    y_pred   = model.predict(X_test)
    mae      = mean_absolute_error(y_test, y_pred)
    dir_acc  = float(np.mean(np.sign(y_pred) == np.sign(y_test.values)) * 100)

    next_X           = build_next_features(df, lookback)
    next_X[vol_cols] = scaler.transform(next_X[vol_cols])
    pred_return      = model.predict(next_X)[0]
    last_close       = df["Close"].iloc[-1]
    pred_close       = last_close * (1 + pred_return / 100)
    last_date        = df.index[-1].date()

    coef_df = pd.DataFrame({
        "Fitur": X.columns,
        "Bobot": model.coef_,
    }).sort_values("Bobot", key=abs, ascending=False).reset_index(drop=True)
    coef_df.index += 1
    coef_df["Bobot"] = coef_df["Bobot"].map("{:+.6f}".format)

    return {
        "ticker"      : ticker,
        "last_date"   : last_date,
        "last_close"  : last_close,
        "pred_return" : pred_return,
        "pred_close"  : pred_close,
        "mae"         : mae,
        "dir_acc"     : dir_acc,
        "lookback"    : lookback,
        "alpha"       : alpha,
        "intercept"   : model.intercept_,
        "coef_df"     : coef_df,
        "n_data"      : len(df),
        "n_test"      : len(X_test),
    }


def print_detail(r: dict) -> None:
    direction = "▲" if r["pred_return"] >= 0 else "▼"
    print(f"\n{'═'*55}")
    print(f"  Linear Regression Predictor — {r['ticker']}")
    print(f"{'═'*55}")
    print(f"  Lookback      : {r['lookback']} hari  |  Alpha: {r['alpha']}")
    print(f"  Data tersedia : {r['n_data']} baris   |  Test : {r['n_test']} baris")
    print(f"  MAE           : {r['mae']:.4f}%       |  Akurasi arah: {r['dir_acc']:.1f}%")
    print()
    print(f"  Bobot (koefisien) per fitur:")
    print(f"  {r['coef_df'].to_string()}")
    print(f"  Intercept     : {r['intercept']:+.6f}")
    print()
    print(f"  ── Prediksi Besok ──────────────────────────────")
    print(f"  Data terakhir : {r['last_date']}")
    print(f"  Last Close    : IDR {r['last_close']:,.0f}")
    print(f"  DayReturn est : {r['pred_return']:+.2f}%  {direction}")
    print(f"  Est. Close    : IDR {r['pred_close']:,.0f}")
    print(f"{'═'*55}\n")


def run_backtest(ticker: str, n_days: int,
                 lookback_override: int | None = None,
                 alpha_override: float | None = None) -> None:
    ticker = ticker.strip().upper()
    cfg    = load_config(ticker)

    lookback = lookback_override if lookback_override is not None else cfg["lookback"]
    alpha    = alpha_override    if alpha_override    is not None else cfg["alpha"]

    df = load_data(ticker)
    if df.empty:
        print(f"  Ticker '{ticker}' tidak ditemukan di database.")
        return

    X_all, y_all = build_features(df, lookback)
    if len(X_all) < lookback + n_days + 10:
        print(f"  Data tidak cukup untuk backtest {n_days} hari.")
        return

    vol_cols = [c for c in X_all.columns if "volume" in c]
    rows = []

    for i in range(n_days, 0, -1):
        # Gunakan data sampai hari ke -(i+1), prediksi hari ke -i
        X_train = X_all.iloc[:-i].copy()
        y_train = y_all.iloc[:-i]

        scaler = StandardScaler()
        X_train[vol_cols] = scaler.fit_transform(X_train[vol_cols])

        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)

        # Fitur untuk hari yang diprediksi
        X_pred = X_all.iloc[[-i]].copy()
        X_pred[vol_cols] = scaler.transform(X_pred[vol_cols])

        pred_return   = model.predict(X_pred)[0]
        actual_return = y_all.iloc[-i]
        pred_date     = y_all.index[-i].date()
        last_close    = df["Close"].iloc[-(i + 1)]
        actual_close  = df["Close"].iloc[-i]
        pred_close    = last_close * (1 + pred_return / 100)
        error         = pred_return - actual_return
        correct_dir   = np.sign(pred_return) == np.sign(actual_return)

        rows.append({
            "Tanggal"    : pred_date,
            "Last Close" : last_close,
            "Pred Return": pred_return,
            "Act Return" : actual_return,
            "Error"      : error,
            "Pred Close" : pred_close,
            "Act Close"  : actual_close,
            "Arah"       : "✓" if correct_dir else "✗",
        })

    result_df = pd.DataFrame(rows)
    mae      = result_df["Error"].abs().mean()
    dir_acc  = (result_df["Arah"] == "✓").mean() * 100

    print(f"\n{'═'*80}")
    print(f"  Backtest {ticker} — {n_days} hari terakhir  (lookback={lookback}, alpha={alpha})")
    print(f"{'═'*80}")
    print(f"  {'Tanggal':<12} {'Last Close':>11} {'Pred Ret':>9} {'Act Ret':>9} "
          f"{'Error':>8} {'Pred Close':>11} {'Act Close':>10} {'Dir':>4}")
    print(f"  {'─'*12} {'─'*11} {'─'*9} {'─'*9} {'─'*8} {'─'*11} {'─'*10} {'─'*4}")
    for _, r in result_df.iterrows():
        print(f"  {str(r['Tanggal']):<12} {r['Last Close']:>11,.0f} "
              f"{r['Pred Return']:>+8.2f}% {r['Act Return']:>+8.2f}% "
              f"{r['Error']:>+7.2f}% {r['Pred Close']:>11,.0f} "
              f"{r['Act Close']:>10,.0f} {r['Arah']:>4}")
    print(f"{'─'*80}")
    print(f"  MAE: {mae:.4f}%   |   Akurasi arah: {dir_acc:.1f}%   |   "
          f"Benar: {int(dir_acc * n_days / 100)}/{n_days} hari")
    print(f"{'═'*80}\n")


def print_summary(results: list[dict]) -> None:
    df = pd.DataFrame([{
        "Ticker"    : r["ticker"],
        "Last Close": r["last_close"],
        "Est. Close": r["pred_close"],
        "Return Est": r["pred_return"],
        "Arah"      : "▲" if r["pred_return"] >= 0 else "▼",
        "MAE"       : r["mae"],
        "DirAcc"    : r["dir_acc"],
        "Lookback"  : r["lookback"],
    } for r in results]).sort_values("Return Est", ascending=False)

    print(f"\n{'═'*75}")
    print(f"  PREDIKSI BESOK — {date.today()}  (Linear Regression)")
    print(f"{'═'*75}")
    print(f"  {'Ticker':<7} {'Last Close':>11} {'Est. Close':>11} {'Return Est':>11} {'Arah':>4}  {'MAE':>7}  {'DirAcc':>7}  {'LB':>3}")
    print(f"  {'─'*7} {'─'*11} {'─'*11} {'─'*11} {'─'*4}  {'─'*7}  {'─'*7}  {'─'*3}")
    for _, row in df.iterrows():
        print(f"  {row['Ticker']:<7} {row['Last Close']:>11,.0f} {row['Est. Close']:>11,.0f} "
              f"{row['Return Est']:>+10.2f}% {row['Arah']:>4}  {row['MAE']:>6.2f}%  "
              f"{row['DirAcc']:>6.1f}%  {int(row['Lookback']):>3}")
    print(f"{'═'*75}")
    print(f"  Diurutkan: potensi kenaikan tertinggi → terendah")
    print(f"  MAE = error rata-rata prediksi return pada data test historis")
    print(f"  DirAcc = % prediksi arah (naik/turun) yang benar  |  LB = lookback")
    print(f"{'═'*75}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="ridge_predictor.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "BEI Linear Regression Predictor\n"
            "=================================\n"
            "Memprediksi DayReturn_Pct hari berikutnya menggunakan Ridge regression\n"
            "dengan fitur lag DayReturn_Pct dan Volume. Config optimal per ticker\n"
            "dimuat otomatis dari ridge_configs.json jika tersedia."
        ),
        epilog=(
            "Alur kerja yang disarankan:\n"
            "  1. Cari config optimal : python ridge_config_search.py --ticker DMAS\n"
            "  2. Jalankan prediksi   : python ridge_predictor.py --ticker DMAS\n"
            "  3. Semua watchlist     : python ridge_predictor.py --all\n"
            "\n"
            "Contoh:\n"
            "  python ridge_predictor.py --ticker DMAS\n"
            "  python ridge_predictor.py --all\n"
            "  python ridge_predictor.py --all --detail\n"
            "  python ridge_predictor.py --ticker ANTM --lookback 7 --alpha 0.1\n"
        ),
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--ticker", help="Satu kode saham IDX, contoh: DMAS")
    group.add_argument("--all",    action="store_true",
                       help=f"Prediksi semua ticker di {WATCHLIST}")
    parser.add_argument("--lookback", type=int,   default=None,
                        help="Override lookback (hanya berlaku dengan --ticker)")
    parser.add_argument("--alpha",    type=float, default=None,
                        help="Override alpha Ridge (hanya berlaku dengan --ticker)")
    parser.add_argument("--detail",   action="store_true",
                        help="Tampilkan bobot dan detail per ticker (berlaku dengan --all)")
    parser.add_argument("--backtest", type=int, default=None, metavar="N",
                        help="Uji mundur N hari terakhir: prediksi vs aktual (hanya --ticker)")
    args = parser.parse_args()

    if args.backtest is not None:
        if not args.ticker:
            parser.error("--backtest hanya bisa digunakan dengan --ticker")
        run_backtest(args.ticker, args.backtest, args.lookback, args.alpha)
    elif args.all:
        tickers = load_watchlist()
        results = []
        for t in tickers:
            print(f"  Memproses {t}...", end=" ", flush=True)
            r = predict_ticker(t)
            if r:
                results.append(r)
                direction = "▲" if r["pred_return"] >= 0 else "▼"
                print(f"{r['pred_return']:+.2f}% {direction}")
            else:
                print("SKIP (tidak ada data)")
        if results:
            if args.detail:
                for r in results:
                    print_detail(r)
            print_summary(results)
    else:
        r = predict_ticker(args.ticker, args.lookback, args.alpha)
        if r:
            print_detail(r)
        else:
            print(f"  Ticker '{args.ticker}' tidak ditemukan di database.")
