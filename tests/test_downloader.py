"""
Unit tests untuk bei_stock_downloader.py
"""
import sqlite3
from pathlib import Path

import pandas as pd
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from bei_stock_downloader import (
    init_db, save, get_ticker, get_date_range, list_tickers, to_yf_ticker, COLS
)


# ── to_yf_ticker ─────────────────────────────────────────────────────────────

class TestToYfTicker:
    def test_bare_ticker_gets_jk_suffix(self):
        assert to_yf_ticker("BBCA") == "BBCA.JK"

    def test_lowercase_is_uppercased(self):
        assert to_yf_ticker("bbca") == "BBCA.JK"

    def test_index_ticker_passthrough(self):
        assert to_yf_ticker("^JKSE") == "^JKSE"

    def test_already_has_jk_suffix(self):
        assert to_yf_ticker("BBCA.JK") == "BBCA.JK"

    def test_whitespace_stripped(self):
        assert to_yf_ticker("  TLKM  ") == "TLKM.JK"

    def test_various_tickers(self):
        for t in ["ANTM", "GOTO", "TLKM"]:
            assert to_yf_ticker(t) == f"{t}.JK"


# ── init_db ──────────────────────────────────────────────────────────────────

class TestInitDb:
    def test_creates_table(self, tmp_path):
        db = str(tmp_path / "test.db")
        init_db(db)
        conn = sqlite3.connect(db)
        tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        conn.close()
        assert ("daily_prices",) in tables

    def test_idempotent(self, tmp_path):
        db = str(tmp_path / "test.db")
        init_db(db)
        init_db(db)  # panggil dua kali, tidak boleh error

    def test_creates_index(self, tmp_path):
        db = str(tmp_path / "test.db")
        init_db(db)
        conn = sqlite3.connect(db)
        indexes = conn.execute("SELECT name FROM sqlite_master WHERE type='index'").fetchall()
        conn.close()
        names = [i[0] for i in indexes]
        assert "idx_ticker_date" in names

    def test_unique_constraint_exists(self, tmp_path):
        db = str(tmp_path / "test.db")
        init_db(db)
        conn = sqlite3.connect(db)
        # Insert duplicate (Ticker, Date) → harus replace, bukan error
        conn.execute(
            "INSERT INTO daily_prices (Ticker, Date, Close) VALUES ('X', '2024-01-01', 100)"
        )
        conn.execute(
            "INSERT OR REPLACE INTO daily_prices (Ticker, Date, Close) VALUES ('X', '2024-01-01', 200)"
        )
        conn.commit()
        count = conn.execute("SELECT COUNT(*) FROM daily_prices").fetchone()[0]
        conn.close()
        assert count == 1


# ── save & query helpers ──────────────────────────────────────────────────────

def _make_df(ticker: str = "TEST", n: int = 5) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.DataFrame({
        "Ticker"        : ticker,
        "Date"          : dates.strftime("%Y-%m-%d"),
        "Open"          : [100.0] * n,
        "High"          : [105.0] * n,
        "Low"           : [98.0]  * n,
        "Close"         : [102.0] * n,
        "Volume"        : [1_000_000] * n,
        "Frequency"     : [None] * n,
        "GainLoss_IDR"  : [2.0] * n,
        "GainLoss_Pct"  : [2.0] * n,
        "DayReturn_Pct" : [1.0] * n,
        "IntraDay_Range": [7.0] * n,
        "UpdatedAt"     : ["2024-01-01 00:00:00"] * n,
    })


class TestSave:
    def test_rows_inserted(self, tmp_path):
        db = str(tmp_path / "test.db")
        init_db(db)
        save(_make_df("BBCA", 5), db_path=db)
        conn = sqlite3.connect(db)
        count = conn.execute("SELECT COUNT(*) FROM daily_prices WHERE Ticker='BBCA'").fetchone()[0]
        conn.close()
        assert count == 5

    def test_upsert_is_idempotent(self, tmp_path):
        db = str(tmp_path / "test.db")
        init_db(db)
        save(_make_df("BBCA", 5), db_path=db)
        save(_make_df("BBCA", 5), db_path=db)
        conn = sqlite3.connect(db)
        count = conn.execute("SELECT COUNT(*) FROM daily_prices WHERE Ticker='BBCA'").fetchone()[0]
        conn.close()
        assert count == 5

    def test_multiple_tickers(self, tmp_path):
        db = str(tmp_path / "test.db")
        init_db(db)
        save(_make_df("BBCA", 3), db_path=db)
        save(_make_df("ANTM", 4), db_path=db)
        conn = sqlite3.connect(db)
        count = conn.execute("SELECT COUNT(*) FROM daily_prices").fetchone()[0]
        conn.close()
        assert count == 7


class TestGetTicker:
    def test_returns_correct_ticker(self, tmp_path):
        db = str(tmp_path / "test.db")
        init_db(db)
        save(_make_df("BBCA", 3), db_path=db)
        df = get_ticker("BBCA", db_path=db)
        assert len(df) == 3
        assert (df["Ticker"] == "BBCA").all()

    def test_sorted_ascending(self, tmp_path):
        db = str(tmp_path / "test.db")
        init_db(db)
        save(_make_df("BBCA", 5), db_path=db)
        df = get_ticker("BBCA", db_path=db)
        assert list(df["Date"]) == sorted(df["Date"].tolist())

    def test_unknown_ticker_returns_empty(self, tmp_path):
        db = str(tmp_path / "test.db")
        init_db(db)
        df = get_ticker("XXXX", db_path=db)
        assert df.empty


class TestGetDateRange:
    def test_filters_by_date(self, tmp_path):
        db = str(tmp_path / "test.db")
        init_db(db)
        save(_make_df("BBCA", 10), db_path=db)
        df = get_date_range("BBCA", "2024-01-01", "2024-01-05", db_path=db)
        assert len(df) > 0
        assert all(df["Date"] >= "2024-01-01")
        assert all(df["Date"] <= "2024-01-05")

    def test_out_of_range_returns_empty(self, tmp_path):
        db = str(tmp_path / "test.db")
        init_db(db)
        save(_make_df("BBCA", 5), db_path=db)
        df = get_date_range("BBCA", "2099-01-01", "2099-12-31", db_path=db)
        assert df.empty


class TestListTickers:
    def test_returns_all_tickers(self, tmp_path):
        db = str(tmp_path / "test.db")
        init_db(db)
        save(_make_df("BBCA", 2), db_path=db)
        save(_make_df("ANTM", 2), db_path=db)
        tickers = list_tickers(db_path=db)
        assert set(tickers) == {"BBCA", "ANTM"}

    def test_sorted_alphabetically(self, tmp_path):
        db = str(tmp_path / "test.db")
        init_db(db)
        save(_make_df("TLKM", 1), db_path=db)
        save(_make_df("ANTM", 1), db_path=db)
        save(_make_df("BBCA", 1), db_path=db)
        tickers = list_tickers(db_path=db)
        assert tickers == sorted(tickers)

    def test_empty_db_returns_empty_list(self, tmp_path):
        db = str(tmp_path / "test.db")
        init_db(db)
        assert list_tickers(db_path=db) == []
