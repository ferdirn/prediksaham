"""
BEI Logistic Regression Classifier
=====================================
Memprediksi ARAH (naik/turun) harga saham hari berikutnya.
Config optimal per ticker dimuat dari logistic_configs.json.

Fitur: ret_lag, volchg_lag, momentum_3/5, abs_ret, jkse_lag1
Target: 1 = besok naik, 0 = besok turun

Usage:
    python logistic_classifier.py --ticker DMAS
    python logistic_classifier.py --all
    python logistic_classifier.py --ticker DMAS --backtest 30
    python logistic_classifier.py --all --detail
"""

import argparse
import sqlite3
from datetime import date
from pathlib import Path

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from utils import load_watchlist, load_config_json

SQLITE_PATH    = "bei_stocks.db"
LR_CLS_CONFIGS = "logistic_configs.json"
WATCHLIST      = "watchlist.txt"

DEFAULT_CONFIG = {"lookback": 3, "C": 1.0}


def load_config(ticker: str) -> dict:
    return load_config_json(LR_CLS_CONFIGS, ticker, DEFAULT_CONFIG.copy())


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


def load_jkse() -> pd.Series:
    conn = sqlite3.connect(SQLITE_PATH)
    df   = pd.read_sql_query(
        "SELECT Date, DayReturn_Pct FROM daily_prices WHERE Ticker = '^JKSE' ORDER BY Date ASC",
        conn
    )
    conn.close()
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    return df["DayReturn_Pct"].rename("jkse")


def build_features(df: pd.DataFrame, jkse: pd.Series,
                   lookback: int) -> tuple[pd.DataFrame, pd.Series]:
    feat = {}

    for lag in range(1, lookback + 1):
        feat[f"ret_lag{lag}"] = df["DayReturn_Pct"].shift(lag)

    vol_chg = df["Volume"].replace(0, np.nan).pct_change() * 100
    for lag in range(1, lookback + 1):
        feat[f"volchg_lag{lag}"] = vol_chg.shift(lag)

    feat["momentum_3"]  = df["DayReturn_Pct"].shift(1).rolling(3).sum()
    feat["momentum_5"]  = df["DayReturn_Pct"].shift(1).rolling(5).sum()
    feat["abs_ret_lag1"] = df["DayReturn_Pct"].shift(1).abs()
    feat["jkse_lag1"]   = jkse.shift(1).reindex(df.index)

    feat_df = pd.DataFrame(feat, index=df.index)
    target  = (df["DayReturn_Pct"] > 0).astype(int)
    feat_df["target"] = target
    feat_df = feat_df[df["DayReturn_Pct"] != 0]  # singkirkan hari flat (data artifact)
    feat_df.dropna(inplace=True)

    X = feat_df.drop(columns=["target"])
    y = feat_df["target"]
    return X, y


def build_next_features(df: pd.DataFrame, jkse: pd.Series,
                        lookback: int, feature_cols: list[str]) -> pd.DataFrame:
    returns  = df["DayReturn_Pct"].values
    volumes  = df["Volume"].values
    row: dict = {}

    for lag in range(1, lookback + 1):
        row[f"ret_lag{lag}"] = returns[-lag]

    vol_chg = pd.Series(volumes).replace(0, np.nan).pct_change().values * 100
    for lag in range(1, lookback + 1):
        row[f"volchg_lag{lag}"] = vol_chg[-lag] if lag <= len(vol_chg) else np.nan

    row["momentum_3"]   = float(np.nansum(returns[-3:]))
    row["momentum_5"]   = float(np.nansum(returns[-5:]))
    row["abs_ret_lag1"] = abs(returns[-1])

    jkse_vals = jkse.reindex(df.index).ffill()
    last_jkse = jkse_vals.iloc[-1]
    row["jkse_lag1"] = float(last_jkse) if pd.notna(last_jkse) else 0.0

    return pd.DataFrame([row])[feature_cols]


def train_model(X: pd.DataFrame, y: pd.Series,
                c: float) -> tuple[LogisticRegression, StandardScaler]:
    scaler  = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    model   = LogisticRegression(C=c, class_weight="balanced",
                                 max_iter=1000, random_state=42)
    model.fit(X_scaled, y)
    return model, scaler


def predict_ticker(ticker: str) -> dict | None:
    ticker = ticker.strip().upper()
    cfg    = load_config(ticker)
    lookback, c = cfg["lookback"], cfg["C"]

    df   = load_data(ticker)
    jkse = load_jkse()

    df = df[df["DayReturn_Pct"].notna()]
    if df.empty or len(df) < lookback + 10:
        return None

    X, y = build_features(df, jkse, lookback)
    model, scaler = train_model(X, y, c)

    next_X       = build_next_features(df, jkse, lookback, list(X.columns))
    next_X_scaled = pd.DataFrame(scaler.transform(next_X), columns=next_X.columns)
    prob_up      = model.predict_proba(next_X_scaled)[0][1]
    pred_dir     = 1 if prob_up >= 0.5 else 0

    last_close = df["Close"].iloc[-1]
    last_date  = df.index[-1].date()

    coef_df = pd.DataFrame({
        "Fitur": X.columns,
        "Bobot": model.coef_[0],
    }).sort_values("Bobot", key=abs, ascending=False).reset_index(drop=True)
    coef_df.index += 1
    coef_df["Bobot"] = coef_df["Bobot"].map("{:+.4f}".format)

    return {
        "ticker"    : ticker,
        "last_date" : last_date,
        "last_close": last_close,
        "prob_up"   : prob_up,
        "pred_dir"  : pred_dir,
        "lookback"  : lookback,
        "C"         : c,
        "coef_df"   : coef_df,
        "n_data"    : len(df),
        "X"         : X,
        "y"         : y,
    }


def print_detail(r: dict) -> None:
    arrow      = "▲ NAIK" if r["pred_dir"] == 1 else "▼ TURUN"
    confidence = r["prob_up"] if r["pred_dir"] == 1 else 1 - r["prob_up"]
    print(f"\n{'═'*55}")
    print(f"  Classifier — {r['ticker']}")
    print(f"{'═'*55}")
    print(f"  Lookback : {r['lookback']} hari  |  C: {r['C']}")
    print(f"  Data     : {r['n_data']} baris")
    print()
    print(f"  Bobot (koefisien) per fitur:")
    print(f"  {r['coef_df'].to_string()}")
    print()
    print(f"  ── Prediksi Besok ──────────────────────────")
    print(f"  Data terakhir : {r['last_date']}")
    print(f"  Last Close    : IDR {r['last_close']:,.0f}")
    print(f"  Prediksi arah : {arrow}")
    print(f"  Prob naik     : {r['prob_up']*100:.1f}%")
    print(f"  Confidence    : {confidence*100:.1f}%")
    print(f"{'═'*55}\n")


def run_backtest(ticker: str, n_days: int, print_detail_rows: bool = True) -> dict | None:
    ticker = ticker.strip().upper()
    cfg    = load_config(ticker)
    lookback, c = cfg["lookback"], cfg["C"]

    df   = load_data(ticker)
    jkse = load_jkse()
    df   = df[df["DayReturn_Pct"].notna()]

    X_all, y_all = build_features(df, jkse, lookback)
    if len(X_all) < lookback + n_days + 20:
        print(f"  [{ticker}] Data tidak cukup untuk backtest {n_days} hari.")
        return None

    rows = []
    for i in range(n_days, 0, -1):
        X_train = X_all.iloc[:-i]
        y_train = y_all.iloc[:-i]

        if len(X_train) < 30:
            continue

        model, scaler = train_model(X_train, y_train, c)

        X_pred    = pd.DataFrame(scaler.transform(X_all.iloc[[-i]]), columns=X_all.columns)
        prob_up   = model.predict_proba(X_pred)[0][1]
        pred      = 1 if prob_up >= 0.5 else 0
        actual    = int(y_all.iloc[-i])
        ret_act   = df["DayReturn_Pct"].loc[y_all.index[-i]]
        pred_date = y_all.index[-i].date()

        rows.append({
            "Tanggal"  : pred_date,
            "Prob Naik": prob_up * 100,
            "Prediksi" : "▲" if pred == 1 else "▼",
            "Aktual"   : "▲" if actual == 1 else "▼",
            "Ret Act"  : ret_act,
            "Benar"    : "✓" if pred == actual else "✗",
        })

    if not rows:
        return None

    res_df  = pd.DataFrame(rows)
    acc     = (res_df["Benar"] == "✓").mean() * 100
    n_benar = int((res_df["Benar"] == "✓").sum())

    if print_detail_rows:
        print(f"\n{'═'*70}")
        print(f"  Backtest {ticker} — {n_days} hari  (lookback={lookback}, C={c})")
        print(f"{'═'*70}")
        print(f"  {'Tanggal':<12} {'Prob Naik':>9} {'Pred':>5} {'Aktual':>7} {'Ret Act':>8} {'Benar':>6}")
        print(f"  {'─'*12} {'─'*9} {'─'*5} {'─'*7} {'─'*8} {'─'*6}")
        for _, row in res_df.iterrows():
            print(f"  {str(row['Tanggal']):<12} {row['Prob Naik']:>8.1f}%  "
                  f"{row['Prediksi']:>5} {row['Aktual']:>7} "
                  f"{row['Ret Act']:>+7.2f}%  {row['Benar']:>5}")
        print(f"{'─'*70}")
        print(f"  Akurasi arah: {acc:.1f}%   |   Benar: {n_benar}/{len(rows)} hari")
        print(f"{'═'*70}\n")

    return {
        "ticker"  : ticker,
        "acc"     : acc,
        "n_benar" : n_benar,
        "n_total" : len(rows),
        "lookback": lookback,
        "C"       : c,
    }


def print_backtest_summary(results: list[dict], n_days: int) -> None:
    df = pd.DataFrame(results).sort_values("acc", ascending=False)

    print(f"\n{'═'*60}")
    print(f"  RINGKASAN BACKTEST — {n_days} hari  (Logistic Regression)")
    print(f"{'═'*60}")
    print(f"  {'Ticker':<8} {'Akurasi':>8} {'Benar':>8} {'Total':>6} {'LB':>4} {'C':>7}")
    print(f"  {'─'*8} {'─'*8} {'─'*8} {'─'*6} {'─'*4} {'─'*7}")
    for _, r in df.iterrows():
        bar = "█" * int(r["acc"] / 5)
        print(f"  {r['ticker']:<8} {r['acc']:>7.1f}% {r['n_benar']:>5}/{r['n_total']:<5}"
              f" {int(r['lookback']):>4} {r['C']:>7}  {bar}")
    avg = df["acc"].mean()
    print(f"  {'─'*60}")
    print(f"  {'Rata-rata':<8} {avg:>7.1f}%")
    print(f"{'═'*60}\n")


def print_summary(results: list[dict]) -> None:
    rows = []
    for r in results:
        confidence = r["prob_up"] if r["pred_dir"] == 1 else 1 - r["prob_up"]
        rows.append({
            "Ticker"    : r["ticker"],
            "Last Close": r["last_close"],
            "Prediksi"  : "▲ NAIK" if r["pred_dir"] == 1 else "▼ TURUN",
            "Prob Naik" : r["prob_up"] * 100,
            "Confidence": confidence * 100,
            "Lookback"  : r["lookback"],
        })

    df = pd.DataFrame(rows).sort_values("Confidence", ascending=False)

    print(f"\n{'═'*65}")
    print(f"  PREDIKSI ARAH BESOK — {date.today()}  (Logistic Regression)")
    print(f"{'═'*65}")
    print(f"  {'Ticker':<7} {'Last Close':>11} {'Prediksi':>9} {'Prob Naik':>10} {'Confidence':>11} {'LB':>3}")
    print(f"  {'─'*7} {'─'*11} {'─'*9} {'─'*10} {'─'*11} {'─'*3}")
    for _, row in df.iterrows():
        print(f"  {row['Ticker']:<7} {row['Last Close']:>11,.0f} {row['Prediksi']:>9} "
              f"{row['Prob Naik']:>9.1f}% {row['Confidence']:>10.1f}%  {int(row['Lookback']):>2}")
    print(f"{'═'*65}")
    print(f"  Confidence = seberapa yakin model terhadap prediksinya")
    print(f"  LB = lookback  |  Diurutkan: confidence tertinggi → terendah")
    print(f"{'═'*65}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="logistic_classifier.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "BEI Logistic Regression Classifier\n"
            "=====================================\n"
            "Memprediksi ARAH (naik/turun) saham hari berikutnya.\n"
            "Config optimal dari logistic_configs.json (jalankan logistic_config_search.py dulu)."
        ),
        epilog=(
            "Alur kerja:\n"
            "  1. Riset config : python logistic_config_search.py\n"
            "  2. Prediksi     : python logistic_classifier.py --all\n"
            "  3. Backtest     : python logistic_classifier.py --ticker DMAS --backtest 30\n"
            "\n"
            "Contoh:\n"
            "  python logistic_classifier.py --ticker DMAS\n"
            "  python logistic_classifier.py --all\n"
            "  python logistic_classifier.py --all --detail\n"
            "  python logistic_classifier.py --ticker DMAS --backtest 30\n"
        ),
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--ticker", help="Satu kode saham IDX")
    group.add_argument("--all",    action="store_true",
                       help="Prediksi semua ticker di watchlist.txt")
    parser.add_argument("--backtest", type=int, default=None, metavar="N",
                        help="Backtest N hari terakhir (berlaku untuk --ticker maupun --all)")
    parser.add_argument("--detail",   action="store_true",
                        help="Tampilkan baris per hari pada backtest --all, atau bobot pada prediksi --all")
    args = parser.parse_args()

    if args.all:
        tickers = load_watchlist()
        if args.backtest:
            bt_results = []
            for t in tickers:
                print(f"  Backtest {t}...", end=" ", flush=True)
                r = run_backtest(t, args.backtest, print_detail_rows=args.detail)
                if r:
                    bt_results.append(r)
                    print(f"akurasi {r['acc']:.1f}%  ({r['n_benar']}/{r['n_total']})")
                else:
                    print("SKIP")
            if bt_results:
                print_backtest_summary(bt_results, args.backtest)
        else:
            results = []
            for t in tickers:
                print(f"  Memproses {t}...", end=" ", flush=True)
                r = predict_ticker(t)
                if r:
                    results.append(r)
                    arrow = "▲" if r["pred_dir"] == 1 else "▼"
                    conf  = r["prob_up"] if r["pred_dir"] == 1 else 1 - r["prob_up"]
                    print(f"{arrow}  confidence {conf*100:.1f}%")
                else:
                    print("SKIP")
            if results:
                if args.detail:
                    for r in results:
                        print_detail(r)
                print_summary(results)
    elif args.backtest:
        run_backtest(args.ticker, args.backtest, print_detail_rows=True)
    else:
        r = predict_ticker(args.ticker)
        if r:
            print_detail(r)
        else:
            print(f"  Ticker '{args.ticker}' tidak ditemukan.")
