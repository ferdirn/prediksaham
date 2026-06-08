"""
Unit tests untuk lstm_predictor.py — fungsi-fungsi pure (tanpa training LSTM).
Training LSTM tidak di-test di sini karena terlalu lambat untuk unit test.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import sys, os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.path.insert(0, str(Path(__file__).parent.parent))

from lstm_predictor import make_sequences, add_indicators, load_ticker_config


# ── make_sequences ────────────────────────────────────────────────────────────

class TestMakeSequences:
    def test_output_shapes(self):
        data = np.random.randn(50, 4).astype(np.float32)
        X, y = make_sequences(data, lookback=10, forecast=1)
        assert X.shape == (40, 10, 4)
        assert y.shape == (40, 1)

    def test_forecast_gt_1(self):
        data = np.random.randn(50, 4).astype(np.float32)
        X, y = make_sequences(data, lookback=5, forecast=3)
        expected_samples = 50 - 5 - 3 + 1
        assert X.shape[0] == expected_samples
        assert y.shape[1] == 3

    def test_lookback_1(self):
        data = np.random.randn(20, 2).astype(np.float32)
        X, y = make_sequences(data, lookback=1, forecast=1)
        assert X.shape == (19, 1, 2)

    def test_x_contains_correct_window(self):
        data = np.arange(30).reshape(30, 1).astype(np.float32)
        X, y = make_sequences(data, lookback=3, forecast=1)
        # Sequence pertama harus [0, 1, 2], target [3]
        np.testing.assert_array_equal(X[0, :, 0], [0, 1, 2])
        np.testing.assert_array_equal(y[0], [3])

    def test_y_follows_x(self):
        data = np.arange(20).reshape(20, 1).astype(np.float32)
        X, y = make_sequences(data, lookback=2, forecast=1)
        # y[i] harus = data[i + lookback]
        for i in range(len(y)):
            assert y[i][0] == data[i + 2][0]


# ── add_indicators ────────────────────────────────────────────────────────────

class TestAddIndicators:
    def _make_df(self, n: int = 60) -> pd.DataFrame:
        np.random.seed(10)
        dates = pd.date_range("2024-01-01", periods=n, freq="B")
        close = 1000 + np.cumsum(np.random.randn(n) * 10)
        volume = np.random.randint(50_000_000, 200_000_000, n).astype(float)
        ret = pd.Series(close).pct_change().mul(100).values
        return pd.DataFrame({
            "Open"         : np.round(close * 0.999, 2),
            "High"         : np.round(close * 1.005, 2),
            "Low"          : np.round(close * 0.995, 2),
            "Close"        : np.round(close, 2),
            "Volume"       : volume,
            "GainLoss_Pct" : np.round(np.random.randn(n) * 0.5, 4),
            "DayReturn_Pct": np.round(ret, 4),
        }, index=dates)

    def test_returns_dataframe(self):
        df = self._make_df()
        result = add_indicators(df)
        assert isinstance(result, pd.DataFrame)

    def test_no_extra_columns_by_default(self):
        # add_indicators harus mengembalikan kolom yang sama atau lebih banyak
        df = self._make_df()
        result = add_indicators(df)
        assert len(result.columns) >= len(df.columns)

    def test_index_preserved(self):
        df = self._make_df()
        result = add_indicators(df)
        assert list(result.index) == list(df.index)


# ── load_ticker_config ────────────────────────────────────────────────────────

class TestLoadTickerConfig:
    def test_returns_empty_when_no_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr("lstm_predictor.TICKER_CONFIGS", str(tmp_path / "nofile.json"))
        cfg = load_ticker_config("BBCA")
        assert cfg == {}

    def test_loads_saved_config(self, tmp_path, monkeypatch):
        configs = {"BBCA": {"lookback": 20, "epochs": 200, "seed": 42}}
        p = tmp_path / "lstm_configs.json"
        p.write_text(json.dumps(configs))
        monkeypatch.setattr("lstm_predictor.TICKER_CONFIGS", str(p))
        cfg = load_ticker_config("BBCA")
        assert cfg["lookback"] == 20
        assert cfg["epochs"] == 200

    def test_unknown_ticker_returns_empty(self, tmp_path, monkeypatch):
        configs = {"BBCA": {"lookback": 20}}
        p = tmp_path / "lstm_configs.json"
        p.write_text(json.dumps(configs))
        monkeypatch.setattr("lstm_predictor.TICKER_CONFIGS", str(p))
        cfg = load_ticker_config("XXXX")
        assert cfg == {}
