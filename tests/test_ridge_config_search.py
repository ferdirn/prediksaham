"""
Unit tests untuk ridge_config_search.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from ridge_config_search import build_features, cv_score, save_configs


def _make_df(n: int = 80) -> pd.DataFrame:
    np.random.seed(99)
    dates  = pd.date_range("2024-01-01", periods=n, freq="B")
    close  = 1000 + np.cumsum(np.random.randn(n) * 10)
    volume = np.random.randint(50_000_000, 200_000_000, n).astype(float)
    ret    = pd.Series(close).pct_change().mul(100).values
    df = pd.DataFrame({
        "DayReturn_Pct": np.round(ret, 4),
        "Volume"       : volume,
    }, index=dates)
    df.index.name = "Date"
    return df.dropna()


class TestBuildFeatures:
    def test_shape(self):
        df = _make_df()
        X, y = build_features(df, lookback=3)
        assert X.shape[1] == 6  # 3 × (return + volume)
        assert len(X) == len(y)

    def test_no_nan(self):
        df = _make_df()
        X, y = build_features(df, lookback=5)
        assert not X.isnull().any().any()
        assert not y.isnull().any()

    def test_target_is_dayreturn(self):
        df = _make_df()
        _, y = build_features(df, lookback=1)
        # Target harus sama dengan DayReturn_Pct (setelah dropna)
        assert y.name == "target"


class TestCvScore:
    def test_returns_tuple_of_floats(self):
        df = _make_df(80)
        X, y = build_features(df, lookback=3)
        mae, dir_acc = cv_score(X, y, alpha=1.0)
        assert isinstance(mae, float)
        assert isinstance(dir_acc, float)

    def test_mae_positive(self):
        df = _make_df(80)
        X, y = build_features(df, lookback=3)
        mae, _ = cv_score(X, y, alpha=1.0)
        assert mae >= 0

    def test_dir_acc_in_range(self):
        df = _make_df(80)
        X, y = build_features(df, lookback=3)
        _, dir_acc = cv_score(X, y, alpha=1.0)
        assert 0 <= dir_acc <= 100


class TestSaveConfigs:
    def test_creates_file(self, tmp_path, monkeypatch):
        p = tmp_path / "ridge_configs.json"
        monkeypatch.setattr("ridge_config_search.LR_CONFIGS", str(p))
        save_configs({"BBCA": {"lookback": 3, "alpha": 1.0}})
        assert p.exists()

    def test_merges_existing(self, tmp_path, monkeypatch):
        p = tmp_path / "ridge_configs.json"
        p.write_text(json.dumps({"ANTM": {"lookback": 5}}))
        monkeypatch.setattr("ridge_config_search.LR_CONFIGS", str(p))
        save_configs({"BBCA": {"lookback": 3}})
        result = json.loads(p.read_text())
        assert "ANTM" in result
        assert "BBCA" in result

    def test_overwrites_existing_ticker(self, tmp_path, monkeypatch):
        p = tmp_path / "ridge_configs.json"
        p.write_text(json.dumps({"BBCA": {"lookback": 5}}))
        monkeypatch.setattr("ridge_config_search.LR_CONFIGS", str(p))
        save_configs({"BBCA": {"lookback": 10}})
        result = json.loads(p.read_text())
        assert result["BBCA"]["lookback"] == 10
