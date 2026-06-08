"""
Unit tests untuk stock_viewer.py
"""
import sqlite3
from pathlib import Path

import pandas as pd
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from stock_viewer import peek


def _make_db(tmp_path, ticker: str = "BBCA", n: int = 10) -> str:
    db = str(tmp_path / "test.db")
    conn = sqlite3.connect(db)
    conn.execute("""
        CREATE TABLE daily_prices (
            Ticker TEXT, Date TEXT, Open REAL, High REAL, Low REAL,
            Close REAL, Volume INTEGER, GainLoss_Pct REAL,
            DayReturn_Pct REAL, IntraDay_Range REAL
        )
    """)
    dates = [f"2024-{m:02d}-{d:02d}" for m, d in
             [(1, i+1) for i in range(n)]]
    rows = [(ticker, dates[i], 100.0, 105.0, 98.0, 102.0,
             1_000_000, 2.0, 1.0 if i > 0 else None, 7.0) for i in range(n)]
    conn.executemany(
        "INSERT INTO daily_prices VALUES (?,?,?,?,?,?,?,?,?,?)", rows
    )
    conn.commit()
    conn.close()
    return db


class TestPeek:
    def test_returns_correct_rows(self, tmp_path, monkeypatch, capsys):
        db = _make_db(tmp_path, "BBCA", 10)
        monkeypatch.setattr("stock_viewer.SQLITE_PATH", db)
        peek("BBCA", 5)
        captured = capsys.readouterr()
        assert "BBCA" in captured.out

    def test_unknown_ticker_prints_warning(self, tmp_path, monkeypatch, capsys):
        db = _make_db(tmp_path, "BBCA", 5)
        monkeypatch.setattr("stock_viewer.SQLITE_PATH", db)
        peek("XXXX", 5)
        captured = capsys.readouterr()
        assert "tidak ditemukan" in captured.out

    def test_lowercase_ticker_normalized(self, tmp_path, monkeypatch, capsys):
        db = _make_db(tmp_path, "BBCA", 5)
        monkeypatch.setattr("stock_viewer.SQLITE_PATH", db)
        peek("bbca", 5)
        captured = capsys.readouterr()
        assert "BBCA" in captured.out

    def test_data_sorted_ascending(self, tmp_path, monkeypatch, capsys):
        db = _make_db(tmp_path, "BBCA", 10)
        monkeypatch.setattr("stock_viewer.SQLITE_PATH", db)
        peek("BBCA", 10)
        captured = capsys.readouterr()
        # Tanggal pertama harus muncul sebelum tanggal terakhir
        pos_first = captured.out.find("2024-01-01")
        pos_last  = captured.out.find("2024-01-10")
        assert pos_first < pos_last

    def test_rows_limited_to_n(self, tmp_path, monkeypatch, capsys):
        db = _make_db(tmp_path, "BBCA", 10)
        monkeypatch.setattr("stock_viewer.SQLITE_PATH", db)
        peek("BBCA", 3)
        captured = capsys.readouterr()
        assert "3 data terakhir" in captured.out
