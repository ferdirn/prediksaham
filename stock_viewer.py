"""
BEI Stock Data Viewer
======================
Menampilkan N data terakhir untuk satu ticker dari bei_stocks.db.

Usage:
    python stock_viewer.py BBCA
    python stock_viewer.py BBCA --data 20
    python stock_viewer.py BBCA --data 5

Kolom yang ditampilkan:
    Date, Open, High, Low, Close, Volume, GainLoss_Pct, DayReturn_Pct, IntraDay_Range

Default: 10 baris terakhir, diurutkan ascending (terlama → terbaru).
"""

import argparse
import sqlite3

import pandas as pd

SQLITE_PATH = "bei_stocks.db"


def peek(ticker: str, rows: int) -> None:
    ticker = ticker.strip().upper()

    conn = sqlite3.connect(SQLITE_PATH)
    df = pd.read_sql_query(
        "SELECT Date, Open, High, Low, Close, Volume, GainLoss_Pct, DayReturn_Pct, IntraDay_Range "
        "FROM daily_prices WHERE Ticker = ? ORDER BY Date DESC LIMIT ?",
        conn, params=(ticker, rows)
    )
    conn.close()

    if df.empty:
        print(f"  Ticker '{ticker}' tidak ditemukan di database.")
        return

    df = df.iloc[::-1].reset_index(drop=True)
    df.index += 1

    df["Close"]         = df["Close"].map("{:,.0f}".format)
    df["Open"]          = df["Open"].map("{:,.0f}".format)
    df["High"]          = df["High"].map("{:,.0f}".format)
    df["Low"]           = df["Low"].map("{:,.0f}".format)
    df["Volume"]        = df["Volume"].map("{:,.0f}".format)
    df["GainLoss_Pct"]  = df["GainLoss_Pct"].map("{:+.2f}%".format)
    df["DayReturn_Pct"] = df["DayReturn_Pct"].map(
        lambda x: f"{x:+.2f}%" if pd.notna(x) else "-"
    )
    df["IntraDay_Range"] = df["IntraDay_Range"].map("{:,.0f}".format)

    print(f"\n  {ticker} — {rows} data terakhir")
    print(f"  {'─' * 90}")
    print(df.to_string())
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="stock_viewer.py",
        description="Tampilkan N data terakhir untuk satu ticker dari bei_stocks.db.",
    )
    parser.add_argument("ticker", help="Kode saham IDX, contoh: BBCA")
    parser.add_argument("--data", type=int, default=10, metavar="N",
                        help="Jumlah baris terakhir yang ditampilkan (default: 10)")
    args = parser.parse_args()

    peek(args.ticker, args.data)
