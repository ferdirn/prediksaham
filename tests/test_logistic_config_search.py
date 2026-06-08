"""
Unit tests untuk logistic_config_search.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from logistic_config_search import build_features, cv_score, save_configs


def _make_df(n: int = 100) -> pd.DataFrame:
    np.random.seed(42)
    dates  = pd.date_range("2024-01-01", periods=n, freq="B")
    close  = 1000 + np.cumsum(np.random.randn(n) * 10)
    volume = np.random.randint(50_000_000, 300_000_000, n).astype(float)
    ret    = pd.Series(close).pct_change().mul(100).values
    df = pd.DataFrame({
        "DayReturn_Pct": np.round(ret, 4),
        "Volume"       : volume,
    }, index=dates)
    df.index.name = "Date"
    return df.dropna()


def _make_jkse(index: pd.DatetimeIndex) -> pd.Series:
    np.random.seed(7)
    return pd.Series(np.random.randn(len(index)) * 0.5, index=index, name="jkse")


class TestBuildFeatures:
    def test_target_binary(self):
        df   = _make_df()
        jkse = _make_jkse(df.index)
        _, y = build_features(df, jkse, lookback=3)
        assert set(y.unique()).issubset({0, 1})

    def test_no_nan(self):
        df   = _make_df()
        jkse = _make_jkse(df.index)
        X, y = build_features(df, jkse, lookback=3)
        assert not X.isnull().any().any()

    def test_correct_feature_count(self):
        df   = _make_df()
        jkse = _make_jkse(df.index)
        X, _ = build_features(df, jkse, lookback=2)
        # ret_lag×2, volchg_lag×2, momentum_3, momentum_5, abs_ret_lag1, jkse_lag1 = 9
        assert X.shape[1] == 2 + 2 + 3 + 1


class TestCvScore:
    def test_returns_float(self):
        df   = _make_df(100)
        jkse = _make_jkse(df.index)
        X, y = build_features(df, jkse, lookback=3)
        acc  = cv_score(X, y, c=1.0)
        assert isinstance(acc, float)

    def test_accuracy_in_range(self):
        df   = _make_df(100)
        jkse = _make_jkse(df.index)
        X, y = build_features(df, jkse, lookback=3)
        acc  = cv_score(X, y, c=1.0)
        assert 0 <= acc <= 100


class TestSaveConfigs:
    def test_creates_new_file(self, tmp_path, monkeypatch):
        p = tmp_path / "logistic_configs.json"
        monkeypatch.setattr("logistic_config_search.LR_CLS_CONFIGS", str(p))
        save_configs({"BBCA": {"lookback": 2, "C": 0.01}})
        assert p.exists()
        result = json.loads(p.read_text())
        assert "BBCA" in result

    def test_merges_with_existing(self, tmp_path, monkeypatch):
        p = tmp_path / "logistic_configs.json"
        p.write_text(json.dumps({"ANTM": {"lookback": 7}}))
        monkeypatch.setattr("logistic_config_search.LR_CLS_CONFIGS", str(p))
        save_configs({"DMAS": {"lookback": 1}})
        result = json.loads(p.read_text())
        assert "ANTM" in result
        assert "DMAS" in result

    def test_overwrites_existing_key(self, tmp_path, monkeypatch):
        p = tmp_path / "logistic_configs.json"
        p.write_text(json.dumps({"BBCA": {"lookback": 7, "C": 1.0}}))
        monkeypatch.setattr("logistic_config_search.LR_CLS_CONFIGS", str(p))
        save_configs({"BBCA": {"lookback": 2, "C": 0.01}})
        result = json.loads(p.read_text())
        assert result["BBCA"]["lookback"] == 2
        assert result["BBCA"]["C"] == 0.01
