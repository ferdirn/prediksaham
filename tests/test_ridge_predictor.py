"""
Unit tests untuk ridge_predictor.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from ridge_predictor import (
    build_features, build_next_features, load_config, DEFAULT_CONFIG
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_df(n: int = 50) -> pd.DataFrame:
    np.random.seed(42)
    dates  = pd.date_range("2024-01-01", periods=n, freq="B")
    close  = 1000 + np.cumsum(np.random.randn(n) * 10)
    volume = np.random.randint(50_000_000, 200_000_000, n).astype(float)
    ret    = pd.Series(close).pct_change().mul(100).values
    df = pd.DataFrame({
        "Close"        : np.round(close, 2),
        "Volume"       : volume,
        "DayReturn_Pct": np.round(ret, 4),
    }, index=dates)
    df.index.name = "Date"
    return df.dropna()


# ── build_features ────────────────────────────────────────────────────────────

class TestBuildFeatures:
    def test_output_shape(self):
        df = _make_df(50)
        X, y = build_features(df, lookback=3)
        assert len(X) == len(y)
        assert X.shape[1] == 3 * 2  # 3 lag × 2 fitur (return + volume)

    def test_feature_column_names(self):
        df = _make_df(50)
        X, _ = build_features(df, lookback=2)
        expected = ["return_lag1", "volume_lag1", "return_lag2", "volume_lag2"]
        assert list(X.columns) == expected

    def test_no_nan_in_output(self):
        df = _make_df(50)
        X, y = build_features(df, lookback=5)
        assert not X.isnull().any().any()
        assert not y.isnull().any()

    def test_lookback_1(self):
        df = _make_df(50)
        X, y = build_features(df, lookback=1)
        assert X.shape[1] == 2
        assert len(X) == len(df) - 1  # 1 baris hilang karena lag

    def test_x_and_y_same_index(self):
        df = _make_df(50)
        X, y = build_features(df, lookback=3)
        assert list(X.index) == list(y.index)


# ── build_next_features ───────────────────────────────────────────────────────

class TestBuildNextFeatures:
    def test_returns_single_row(self):
        df = _make_df(50)
        X, _ = build_features(df, lookback=3)
        nf = build_next_features(df, lookback=3)
        assert len(nf) == 1

    def test_column_names_match_training(self):
        df = _make_df(50)
        X, _ = build_features(df, lookback=3)
        nf   = build_next_features(df, lookback=3)
        assert list(nf.columns) == list(X.columns)

    def test_lag1_is_last_row(self):
        df = _make_df(50)
        nf = build_next_features(df, lookback=1)
        assert nf["return_lag1"].iloc[0] == pytest.approx(df["DayReturn_Pct"].iloc[-1])
        assert nf["volume_lag1"].iloc[0] == pytest.approx(df["Volume"].iloc[-1])

    def test_lag2_is_second_to_last(self):
        df = _make_df(50)
        nf = build_next_features(df, lookback=2)
        assert nf["return_lag2"].iloc[0] == pytest.approx(df["DayReturn_Pct"].iloc[-2])


# ── load_config ───────────────────────────────────────────────────────────────

class TestLoadConfig:
    def test_returns_default_when_no_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr("ridge_predictor.LR_CONFIGS", str(tmp_path / "nofile.json"))
        cfg = load_config("BBCA")
        assert cfg == DEFAULT_CONFIG

    def test_loads_ticker_config(self, tmp_path, monkeypatch):
        configs = {"BBCA": {"lookback": 7, "alpha": 10.0}}
        p = tmp_path / "ridge_configs.json"
        p.write_text(json.dumps(configs))
        monkeypatch.setattr("ridge_predictor.LR_CONFIGS", str(p))
        cfg = load_config("BBCA")
        assert cfg["lookback"] == 7
        assert cfg["alpha"] == 10.0

    def test_unknown_ticker_returns_default(self, tmp_path, monkeypatch):
        configs = {"BBCA": {"lookback": 7, "alpha": 10.0}}
        p = tmp_path / "ridge_configs.json"
        p.write_text(json.dumps(configs))
        monkeypatch.setattr("ridge_predictor.LR_CONFIGS", str(p))
        cfg = load_config("XXXX")
        assert cfg == DEFAULT_CONFIG
