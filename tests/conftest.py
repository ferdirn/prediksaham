"""
Shared fixtures untuk semua test.
"""
import sqlite3
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest


# ── Fixtures: sample data ────────────────────────────────────────────────────

@pytest.fixture
def sample_prices_df() -> pd.DataFrame:
    """DataFrame dengan 100 baris data harga saham sintetis, index DatetimeIndex."""
    np.random.seed(42)
    n = 100
    dates = pd.date_range(start="2024-01-01", periods=n, freq="B")
    close = 1000 + np.cumsum(np.random.randn(n) * 10)
    open_ = close * (1 + np.random.randn(n) * 0.002)
    high  = np.maximum(close, open_) * (1 + np.abs(np.random.randn(n) * 0.003))
    low   = np.minimum(close, open_) * (1 - np.abs(np.random.randn(n) * 0.003))
    volume = np.random.randint(50_000_000, 300_000_000, n).astype(float)

    df = pd.DataFrame({
        "Open"          : np.round(open_, 2),
        "High"          : np.round(high, 2),
        "Low"           : np.round(low, 2),
        "Close"         : np.round(close, 2),
        "Volume"        : volume,
        "GainLoss_Pct"  : np.round((close - open_) / open_ * 100, 4),
        "DayReturn_Pct" : pd.Series(close).pct_change().mul(100).round(4).values,
    }, index=dates)
    df.index.name = "Date"
    return df.dropna()


@pytest.fixture
def sample_jkse(sample_prices_df: pd.DataFrame) -> pd.Series:
    """Series DayReturn_Pct untuk ^JKSE dengan index DatetimeIndex yang sama."""
    np.random.seed(7)
    n = len(sample_prices_df)
    returns = np.random.randn(n) * 0.5
    return pd.Series(returns, index=sample_prices_df.index, name="jkse")


@pytest.fixture
def sample_db(tmp_path: pytest.TempPathFactory, sample_prices_df: pd.DataFrame):
    """
    SQLite DB sementara dengan 2 ticker (TEST1, TEST2) + ^JKSE,
    masing-masing 100 baris.
    Mengembalikan path DB sebagai string.
    """
    db_path = str(tmp_path / "test_stocks.db")
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE daily_prices (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            Ticker         TEXT NOT NULL,
            Date           TEXT NOT NULL,
            Open           REAL, High REAL, Low REAL, Close REAL,
            Volume         INTEGER, Frequency INTEGER,
            GainLoss_IDR   REAL, GainLoss_Pct REAL,
            DayReturn_Pct  REAL, IntraDay_Range REAL,
            UpdatedAt      TEXT,
            UNIQUE(Ticker, Date)
        )
    """)
    conn.execute("CREATE INDEX idx_td ON daily_prices (Ticker, Date)")

    for ticker in ["TEST1", "TEST2", "^JKSE"]:
        np.random.seed({"TEST1": 1, "TEST2": 2, "^JKSE": 3}[ticker])
        n = len(sample_prices_df)
        close = 500 + np.cumsum(np.random.randn(n) * 5)
        day_ret = pd.Series(close).pct_change().mul(100).round(4).values

        rows = []
        for i, (idx, row) in enumerate(sample_prices_df.iterrows()):
            rows.append((
                ticker,
                idx.strftime("%Y-%m-%d"),
                round(float(row["Open"]), 2),
                round(float(row["High"]), 2),
                round(float(row["Low"]), 2),
                round(float(close[i]), 2),
                int(row["Volume"]),
                None,
                round(float(close[i] - row["Open"]), 2),
                round(float((close[i] - row["Open"]) / row["Open"] * 100), 4),
                None if np.isnan(day_ret[i]) else round(float(day_ret[i]), 4),
                round(float(row["High"] - row["Low"]), 2),
                "2024-01-01 00:00:00",
            ))
        conn.executemany(
            "INSERT OR REPLACE INTO daily_prices "
            "(Ticker,Date,Open,High,Low,Close,Volume,Frequency,"
            "GainLoss_IDR,GainLoss_Pct,DayReturn_Pct,IntraDay_Range,UpdatedAt) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            rows
        )
    conn.commit()
    conn.close()
    return db_path


@pytest.fixture
def sample_config_file(tmp_path: pytest.TempPathFactory) -> str:
    """Config JSON sementara dengan satu ticker TEST1."""
    import json
    cfg = {
        "TEST1": {"lookback": 3, "alpha": 1.0, "C": 0.1, "cv_mae": 0.8,
                  "cv_dir_acc": 55.0, "data_rows": 100, "last_updated": "2024-01-01"}
    }
    path = tmp_path / "test_configs.json"
    path.write_text(json.dumps(cfg))
    return str(path)
