"""
BEI (Bursa Efek Indonesia) Stock Historical Data Downloader
============================================================
Downloads daily OHLCV + derived metrics for BEI stocks via Yahoo Finance.
Stores data in SQLite (bei_stocks.db).

Usage:
    Default (all watchlist, 5 years): python bei_stock_downloader.py
    Single ticker                    : python bei_stock_downloader.py --ticker BBCA --days 30
    Multiple tickers                 : python bei_stock_downloader.py --ticker BBCA TLKM GOTO --days 14
    From file                        : python bei_stock_downloader.py --file watchlist.txt --days 30
    By years                         : python bei_stock_downloader.py --ticker BBCA --years 3

Defaults:
    --file  watchlist.txt
    --days  1825 (5 years)

Requirements:
    pip install yfinance pandas
"""

import argparse
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf


# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
SQLITE_PATH = "bei_stocks.db"


# ──────────────────────────────────────────────
# SQLITE — INIT
# ──────────────────────────────────────────────

def init_db(db_path: str = SQLITE_PATH):
    """
    Create the daily_prices table if it doesn't exist.

    Schema:
        Ticker         — IDX ticker code (e.g. BBCA)
        Date           — Trading date (YYYY-MM-DD)
        Open           — Opening price in IDR
        High           — Highest price in IDR
        Low            — Lowest price in IDR
        Close          — Closing price in IDR
        Volume         — Total shares traded
        Frequency      — Number of transactions (NULL if unavailable from source)
        GainLoss_IDR   — Close - Open in IDR
        GainLoss_Pct   — (Close - Open) / Open * 100
        DayReturn_Pct  — Close vs previous day's Close (%)
        IntraDay_Range — High - Low in IDR
        UpdatedAt      — Timestamp of last upsert
    """
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS daily_prices (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            Ticker          TEXT    NOT NULL,
            Date            TEXT    NOT NULL,
            Open            REAL,
            High            REAL,
            Low             REAL,
            Close           REAL,
            Volume          INTEGER,
            Frequency       INTEGER,
            GainLoss_IDR    REAL,
            GainLoss_Pct    REAL,
            DayReturn_Pct   REAL,
            IntraDay_Range  REAL,
            UpdatedAt       TEXT,
            UNIQUE(Ticker, Date)
        )
    """)
    # Index for fast ticker + date lookups
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_ticker_date
        ON daily_prices (Ticker, Date)
    """)
    conn.commit()
    conn.close()


# ──────────────────────────────────────────────
# DOWNLOAD
# ──────────────────────────────────────────────

def to_yf_ticker(ticker: str) -> str:
    """Convert bare IDX ticker to Yahoo Finance format (e.g. BBCA -> BBCA.JK).
    Index tickers starting with '^' (e.g. ^JKSE) are passed through unchanged."""
    ticker = ticker.upper().strip()
    if ticker.startswith("^") or ticker.endswith(".JK"):
        return ticker
    return ticker + ".JK"


def download_stock(ticker: str, days: int) -> pd.DataFrame:
    """
    Download historical daily OHLCV data from Yahoo Finance
    and compute derived columns.
    Returns a clean DataFrame ready to be saved to SQLite.
    """
    yf_ticker  = to_yf_ticker(ticker)
    end_date   = datetime.today()
    start_date = end_date - timedelta(days=days)

    print(f"  Downloading {yf_ticker}  ({start_date.date()} → {end_date.date()}) ...")

    raw = yf.download(
        yf_ticker,
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
        interval="1d",
        auto_adjust=True,   # adjusts for stock splits & dividends
        progress=False,
    )

    if raw.empty:
        print(f"  ⚠️  No data for {yf_ticker}. Check ticker symbol or date range.")
        return pd.DataFrame()

    # Flatten MultiIndex columns produced by yfinance >= 0.2
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index.name = "Date"
    df.reset_index(inplace=True)
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")

    # ── Derived columns ──────────────────────────────────────────────────────
    df["GainLoss_IDR"]    = (df["Close"] - df["Open"]).round(2)
    df["GainLoss_Pct"]    = ((df["Close"] - df["Open"]) / df["Open"] * 100).round(4)
    df["DayReturn_Pct"]   = df["Close"].pct_change().mul(100).round(4)   # vs prev close
    df["IntraDay_Range"]  = (df["High"] - df["Low"]).round(2)

    # Frequency: Yahoo Finance free tier does not expose trade-count per day.
    # Leave as NULL. For real frequency data use: RTI Business / Stockbit Pro / IDX API.
    df["Frequency"]  = None

    df["Ticker"]     = ticker.upper().strip().replace(".JK", "")
    df["UpdatedAt"]  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Round OHLC
    for col in ["Open", "High", "Low", "Close"]:
        df[col] = df[col].round(2)

    return df


# ──────────────────────────────────────────────
# SQLITE — SAVE
# ──────────────────────────────────────────────

COLS = [
    "Ticker", "Date", "Open", "High", "Low", "Close",
    "Volume", "Frequency", "GainLoss_IDR", "GainLoss_Pct",
    "DayReturn_Pct", "IntraDay_Range", "UpdatedAt",
]

def save(df: pd.DataFrame, db_path: str = SQLITE_PATH):
    """
    Upsert all rows in df into daily_prices.
    UNIQUE(Ticker, Date) ensures no duplicates — re-running is safe.
    """
    conn = sqlite3.connect(db_path)

    rows = [
        tuple(None if (v is None or (isinstance(v, float) and pd.isna(v))) else v
              for v in (row[c] for c in COLS))
        for _, row in df.iterrows()
    ]

    conn.executemany(
        f"INSERT OR REPLACE INTO daily_prices ({', '.join(COLS)}) "
        f"VALUES ({', '.join(['?'] * len(COLS))})",
        rows
    )
    conn.commit()

    total = conn.execute(
        "SELECT COUNT(*) FROM daily_prices WHERE Ticker = ?",
        (df["Ticker"].iloc[0],)
    ).fetchone()[0]

    conn.close()
    print(f"  ✅ Saved {len(rows)} rows  (total in DB for this ticker: {total})")


# ──────────────────────────────────────────────
# SQLITE — QUERY HELPERS
# ──────────────────────────────────────────────

def get_ticker(ticker: str, db_path: str = SQLITE_PATH) -> pd.DataFrame:
    """Fetch all rows for a ticker, sorted by date ascending."""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        "SELECT * FROM daily_prices WHERE Ticker = ? ORDER BY Date ASC",
        conn, params=(ticker.upper().strip(),)
    )
    conn.close()
    return df


def get_date_range(ticker: str, start: str, end: str, db_path: str = SQLITE_PATH) -> pd.DataFrame:
    """
    Fetch rows for a ticker between two dates (inclusive).
    Dates in 'YYYY-MM-DD' format.
    """
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        "SELECT * FROM daily_prices WHERE Ticker = ? AND Date BETWEEN ? AND ? ORDER BY Date ASC",
        conn, params=(ticker.upper().strip(), start, end)
    )
    conn.close()
    return df


def list_tickers(db_path: str = SQLITE_PATH) -> list:
    """Return list of all tickers stored in the DB."""
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT DISTINCT Ticker FROM daily_prices ORDER BY Ticker"
    ).fetchall()
    conn.close()
    return [r[0] for r in rows]


# ──────────────────────────────────────────────
# MAIN PIPELINE
# ──────────────────────────────────────────────

def process_tickers(tickers: list, days: int):
    init_db()

    for i, raw_ticker in enumerate(tickers, start=1):
        ticker = raw_ticker.upper().strip().replace(".JK", "")
        print(f"\n{'─'*52}")
        print(f"  {i}. {ticker}")
        print(f"{'─'*52}")

        df = download_stock(ticker, days)
        if df.empty:
            continue

        save(df)

        # Preview
        preview_cols = ["Date", "Open", "High", "Low", "Close",
                        "Volume", "GainLoss_IDR", "GainLoss_Pct", "DayReturn_Pct"]
        print(f"\n  Last 5 trading days:")
        print(df[preview_cols].tail(5).to_string(index=False))

    print(f"\n✅ Selesai. Data tersimpan di '{SQLITE_PATH}'")
    print(f"   Tickers di DB: {list_tickers()}")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download data historis saham BEI ke SQLite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Contoh:
  python bei_stock_downloader.py                              # semua watchlist, 5 tahun
  python bei_stock_downloader.py --ticker BBCA --days 14
  python bei_stock_downloader.py --ticker BBCA TLKM GOTO --days 30
  python bei_stock_downloader.py --file watchlist.txt --days 14
  python bei_stock_downloader.py --ticker BBCA --years 3
        """
    )
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--ticker", "--tickers", dest="ticker", type=str, nargs="+",
                       help="Satu atau beberapa ticker, contoh: BBCA atau BBCA TLKM GOTO")
    group.add_argument("--file",    type=str,           help="File teks berisi satu ticker per baris (default: watchlist.txt)")
    parser.add_argument("--days",   type=int, default=1825,
                        help="Jumlah hari ke belakang (default: 1825 / 5 tahun)")
    parser.add_argument("--years",  type=int, default=None,
                        help="Jumlah tahun ke belakang (override --days)")

    args = parser.parse_args()

    if args.years is not None:
        args.days = args.years * 365

    if args.ticker:
        tickers = args.ticker
    else:
        file_path = args.file or "watchlist.txt"
        path = Path(file_path)
        if not path.exists():
            print(f"ERROR: File '{file_path}' tidak ditemukan.")
            exit(1)
        tickers = [
            line.strip() for line in path.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]

    process_tickers(tickers, args.days)
