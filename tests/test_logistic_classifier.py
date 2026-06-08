"""
Unit tests untuk logistic_classifier.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from logistic_classifier import (
    build_features, build_next_features, train_model, load_config, DEFAULT_CONFIG
)


def _make_df(n: int = 100) -> pd.DataFrame:
    np.random.seed(42)
    dates  = pd.date_range("2024-01-01", periods=n, freq="B")
    close  = 1000 + np.cumsum(np.random.randn(n) * 10)
    volume = np.random.randint(50_000_000, 300_000_000, n).astype(float)
    ret    = pd.Series(close).pct_change().mul(100).values
    df = pd.DataFrame({
        "Close"        : np.round(close, 2),
        "Volume"       : volume,
        "DayReturn_Pct": np.round(ret, 4),
    }, index=dates)
    df.index.name = "Date"
    return df.dropna()


def _make_jkse(index: pd.DatetimeIndex) -> pd.Series:
    np.random.seed(7)
    return pd.Series(np.random.randn(len(index)) * 0.5, index=index, name="jkse")


# ── build_features ────────────────────────────────────────────────────────────

class TestBuildFeatures:
    def test_output_shape(self):
        df   = _make_df()
        jkse = _make_jkse(df.index)
        X, y = build_features(df, jkse, lookback=3)
        assert len(X) == len(y)
        # fitur: ret_lag×3, volchg_lag×3, momentum_3, momentum_5, abs_ret_lag1, jkse_lag1
        assert X.shape[1] == 3 + 3 + 3 + 1  # 10 fitur total

    def test_column_names(self):
        df   = _make_df()
        jkse = _make_jkse(df.index)
        X, _ = build_features(df, jkse, lookback=2)
        assert "ret_lag1" in X.columns
        assert "ret_lag2" in X.columns
        assert "volchg_lag1" in X.columns
        assert "momentum_3" in X.columns
        assert "momentum_5" in X.columns
        assert "abs_ret_lag1" in X.columns
        assert "jkse_lag1" in X.columns

    def test_target_is_binary(self):
        df   = _make_df()
        jkse = _make_jkse(df.index)
        _, y = build_features(df, jkse, lookback=3)
        assert set(y.unique()).issubset({0, 1})

    def test_no_nan_in_output(self):
        df   = _make_df()
        jkse = _make_jkse(df.index)
        X, y = build_features(df, jkse, lookback=3)
        assert not X.isnull().any().any()
        assert not y.isnull().any()

    def test_flat_days_excluded(self):
        df   = _make_df()
        jkse = _make_jkse(df.index)
        # Set beberapa hari menjadi flat (return = 0)
        df_flat = df.copy()
        df_flat.iloc[10:15, df_flat.columns.get_loc("DayReturn_Pct")] = 0.0
        X_orig, y_orig = build_features(df, jkse, lookback=3)
        X_flat, y_flat = build_features(df_flat, jkse, lookback=3)
        assert len(y_flat) <= len(y_orig)


# ── build_next_features ───────────────────────────────────────────────────────

class TestBuildNextFeatures:
    def test_returns_single_row(self):
        df   = _make_df()
        jkse = _make_jkse(df.index)
        X, _ = build_features(df, jkse, lookback=3)
        nf   = build_next_features(df, jkse, lookback=3, feature_cols=list(X.columns))
        assert len(nf) == 1

    def test_columns_match_training(self):
        df   = _make_df()
        jkse = _make_jkse(df.index)
        X, _ = build_features(df, jkse, lookback=3)
        nf   = build_next_features(df, jkse, lookback=3, feature_cols=list(X.columns))
        assert list(nf.columns) == list(X.columns)

    def test_abs_ret_is_nonnegative(self):
        df   = _make_df()
        jkse = _make_jkse(df.index)
        X, _ = build_features(df, jkse, lookback=3)
        nf   = build_next_features(df, jkse, lookback=3, feature_cols=list(X.columns))
        assert nf["abs_ret_lag1"].iloc[0] >= 0


# ── train_model ───────────────────────────────────────────────────────────────

class TestTrainModel:
    def test_model_and_scaler_returned(self):
        df   = _make_df()
        jkse = _make_jkse(df.index)
        X, y = build_features(df, jkse, lookback=3)
        model, scaler = train_model(X, y, c=1.0)
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        assert isinstance(model, LogisticRegression)
        assert isinstance(scaler, StandardScaler)

    def test_model_predicts_binary(self):
        df   = _make_df()
        jkse = _make_jkse(df.index)
        X, y = build_features(df, jkse, lookback=3)
        model, scaler = train_model(X, y, c=1.0)
        from sklearn.preprocessing import StandardScaler
        X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns)
        preds = model.predict(X_scaled)
        assert set(preds).issubset({0, 1})

    def test_predict_proba_sums_to_one(self):
        df   = _make_df()
        jkse = _make_jkse(df.index)
        X, y = build_features(df, jkse, lookback=3)
        model, scaler = train_model(X, y, c=1.0)
        from sklearn.preprocessing import StandardScaler
        X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns)
        proba = model.predict_proba(X_scaled)
        assert np.allclose(proba.sum(axis=1), 1.0)


# ── load_config ───────────────────────────────────────────────────────────────

class TestLoadConfig:
    def test_default_when_no_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr("logistic_classifier.LR_CLS_CONFIGS",
                            str(tmp_path / "nofile.json"))
        cfg = load_config("BBCA")
        assert cfg == DEFAULT_CONFIG

    def test_loads_saved_config(self, tmp_path, monkeypatch):
        configs = {"BBCA": {"lookback": 5, "C": 0.01}}
        p = tmp_path / "logistic_configs.json"
        p.write_text(json.dumps(configs))
        monkeypatch.setattr("logistic_classifier.LR_CLS_CONFIGS", str(p))
        cfg = load_config("BBCA")
        assert cfg["lookback"] == 5
        assert cfg["C"] == 0.01

    def test_unknown_ticker_returns_default(self, tmp_path, monkeypatch):
        configs = {"BBCA": {"lookback": 5, "C": 0.01}}
        p = tmp_path / "logistic_configs.json"
        p.write_text(json.dumps(configs))
        monkeypatch.setattr("logistic_classifier.LR_CLS_CONFIGS", str(p))
        cfg = load_config("XXXX")
        assert cfg == DEFAULT_CONFIG
