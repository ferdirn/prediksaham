"""
Unit tests untuk lstm_batch_predict.py — fungsi pure (make_sequences, inverse_close).
Training model tidak di-test karena terlalu lambat untuk unit test.
"""
from pathlib import Path

import numpy as np
import pytest

import sys, os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.path.insert(0, str(Path(__file__).parent.parent))

from lstm_batch_predict import make_sequences, inverse_close
from sklearn.preprocessing import MinMaxScaler


# ── make_sequences ────────────────────────────────────────────────────────────

class TestMakeSequences:
    def test_output_shapes(self):
        data = np.random.randn(50, 3).astype(np.float32)
        X, y = make_sequences(data, lookback=10)
        assert X.shape == (40, 10, 3)
        assert y.shape == (40,)

    def test_y_is_first_feature(self):
        # make_sequences di batch_predict mengambil kolom 0 sebagai target (Close)
        data = np.arange(20).reshape(20, 1).astype(np.float32)
        X, y = make_sequences(data, lookback=3)
        assert y[0] == data[3, 0]

    def test_x_window_correct(self):
        data = np.arange(15).reshape(15, 1).astype(np.float32)
        X, y = make_sequences(data, lookback=3)
        np.testing.assert_array_equal(X[0, :, 0], [0, 1, 2])
        assert y[0] == 3

    def test_multi_feature(self):
        data = np.random.randn(30, 4).astype(np.float32)
        X, y = make_sequences(data, lookback=5)
        assert X.shape == (25, 5, 4)
        assert y.shape == (25,)


# ── inverse_close ─────────────────────────────────────────────────────────────

class TestInverseClose:
    def _make_scaler(self, n_features: int = 4) -> MinMaxScaler:
        np.random.seed(42)
        data = np.random.randn(100, n_features) * 100 + 1000
        scaler = MinMaxScaler()
        scaler.fit(data)
        return scaler

    def test_returns_correct_length(self):
        scaler = self._make_scaler(4)
        vals   = [0.1, 0.5, 0.9]
        result = inverse_close(scaler, vals)
        assert len(result) == 3

    def test_output_in_original_scale(self):
        scaler = self._make_scaler(4)
        # Scaled value 0.0 harus dekat dengan min asli
        result_min = inverse_close(scaler, [0.0])[0]
        result_max = inverse_close(scaler, [1.0])[0]
        assert result_min < result_max

    def test_roundtrip(self):
        scaler = self._make_scaler(4)
        original = np.array([1200.0])
        # Scale Close saja (kolom 0)
        dummy = np.zeros((1, 4))
        dummy[0, 0] = original[0]
        scaled_val = scaler.transform(dummy)[0, 0]
        recovered  = inverse_close(scaler, [scaled_val])[0]
        assert abs(recovered - original[0]) < 1e-3

    def test_accepts_numpy_array(self):
        scaler = self._make_scaler(4)
        vals   = np.array([0.2, 0.6])
        result = inverse_close(scaler, vals)
        assert len(result) == 2
