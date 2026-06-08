# tests/test_io.py

"""Tests for FAST JSON conversion helpers."""

import numpy as np

from fast_python.io import build_json_data


def test_build_json_data_serializes_numpy_values_as_json_arrays_and_scalars():
    """Check native NumPy results are emitted as JSON-compatible values."""

    value = {
        "matrix": np.asarray([[1, 2], [3, 4]], dtype=float),
        "scalar": np.float64(1.5),
    }

    result = build_json_data(value)

    assert result == {
        "matrix": [[1.0, 2.0], [3.0, 4.0]],
        "scalar": 1.5,
    }
