# tests/test_compare.py

"""Tests for FAST parity comparison helpers."""

import numpy as np

from fast_python.compare import compare_json_value
from fast_python.markers import MatlabExpression, MatlabRow


def test_compare_json_value_accepts_numpy_arrays_and_scalars():
    """Check native NumPy values compare against JSON-style values."""

    failures, compared = compare_json_value(
        {
            "matrix": np.asarray([[1, 2], [3, 4]], dtype=float),
            "scalar": np.float64(1.5),
        },
        {
            "matrix": [[1, 2], [3, 4]],
            "scalar": 1.5,
        },
    )

    assert compared > 0
    assert failures == []


def test_compare_json_value_ignores_internal_history_database_subtrees():
    """Check internal database/preprocessing paths do not obscure parity checks."""

    failures, compared = compare_json_value(
        {
            "HistData": {
                "AC": {
                    "A": {
                        "Overview": {
                            "Aisle": object(),
                        }
                    }
                }
            },
            "Specs": {
                "Weight": {
                    "MTOW": 1,
                }
            },
        },
        {
            "HistData": {
                "AC": {
                    "A": {
                        "Overview": {
                            "Aisle": "Single",
                        }
                    }
                }
            },
            "Specs": {
                "Weight": {
                    "MTOW": 1,
                }
            },
        },
    )

    assert compared > 0
    assert failures == []


def test_compare_json_value_compares_marker_repr_with_materialized_value():
    """Check wrapper marker dictionaries compare with native Python values."""

    failures, compared = compare_json_value(
        [[0.0], [1.0]],
        {
            "_python_type": "double",
            "_repr": "[[0.0], [1.0]]",
        },
    )

    assert compared > 0
    assert failures == []


def test_compare_json_value_treats_singleton_columns_as_vectors():
    """Check MATLAB column vectors compare with Python flat vectors."""

    failures, compared = compare_json_value(
        [0.0, 1.0],
        [[0.0], [1.0]],
    )

    assert compared > 0
    assert failures == []


def test_compare_json_value_treats_singleton_rows_as_vectors():
    """Check MATLAB row vectors compare with Python flat vectors."""

    failures, compared = compare_json_value(
        [0.0, 1.0],
        [[0.0, 1.0]],
    )

    assert compared > 0
    assert failures == []


def test_compare_json_value_accepts_wrapper_nan_strings():
    """Check JSON NaN strings compare with Python NaN values."""

    failures, compared = compare_json_value(
        float("nan"),
        "NaN",
    )

    assert compared > 0
    assert failures == []


def test_compare_json_value_compares_input_markers_to_output_markers():
    """Check wrapper input markers do not fail against output marker reprs."""

    failures, compared = compare_json_value(
        {
            "OperDwn": MatlabExpression("@()[1]"),
            "TrnType": MatlabRow([0, 2]),
        },
        {
            "OperDwn": {
                "_python_type": "object",
                "_repr": "<matlab.object object at 0x1>",
            },
            "TrnType": {
                "_python_type": "double",
                "_repr": "[[0.0, 2.0]]",
            },
        },
    )

    assert compared > 0
    assert failures == []


def test_compare_json_value_ignores_missing_default_values():
    """Check default-only fields do not produce missing-key failures."""

    failures, compared = compare_json_value(
        {
            "Specs": {
                "Battery": {
                    "CapCell": float("nan"),
                }
            }
        },
        {
            "Specs": {},
        },
    )

    assert compared > 0
    assert failures == []


def test_compare_json_value_ignores_selected_missing_only_paths():
    """Check unused default paths are ignored only as missing metadata."""

    failures, compared = compare_json_value(
        {
            "Specs": {
                "Power": {
                    "Eta": {
                        "Propeller": 0.8,
                    }
                }
            }
        },
        {
            "Specs": {
                "Power": {
                    "Eta": {}
                }
            }
        },
    )

    assert compared > 0
    assert failures == []


def test_compare_json_value_ignores_missing_internal_profile_function():
    """Check missing wrapper-only callback metadata is ignored."""

    failures, compared = compare_json_value(
        {
            "Mission": {},
        },
        {
            "Mission": {
                "ProfileFxn": {
                    "_python_type": "object",
                    "_repr": "<function AEAProfile at 0x1>",
                }
            },
        },
    )

    assert compared > 0
    assert failures == []


def test_compare_json_value_ignores_prop_arch_split_handles():
    """Check executable split handles do not obscure OutputAircraft parity."""

    failures, compared = compare_json_value(
        {
            "Specs": {
                "Propulsion": {
                    "PropArch": {
                        "OperUps": {
                            "_python_type": "function",
                            "_repr": "<function constant_split at 0x1>",
                        },
                        "OperDwn": {
                            "_python_type": "function",
                            "_repr": "<function constant_split at 0x2>",
                        },
                        "TrnType": [1],
                    }
                }
            }
        },
        {
            "Specs": {
                "Propulsion": {
                    "PropArch": {
                        "OperUps": {
                            "_python_type": "object",
                            "_repr": "<matlab.object object at 0x1>",
                        },
                        "OperDwn": {
                            "_python_type": "object",
                            "_repr": "<matlab.object object at 0x2>",
                        },
                        "TrnType": [1],
                    }
                }
            }
        },
    )

    assert compared > 0
    assert failures == []
