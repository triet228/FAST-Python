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


def test_compare_json_value_accepts_wrapper_rounded_numeric_output():
    """Check saved wrapper JSON precision does not create false parity gaps."""

    failures, compared = compare_json_value(
        40.763830590529125,
        40.7638,
    )

    assert compared > 0
    assert failures == []


def test_compare_json_value_reports_values_outside_wrapper_tolerance():
    """Check meaningful numeric drift still fails the parity comparison."""

    failures, compared = compare_json_value(
        40.7639,
        40.7638,
    )

    assert compared > 0
    assert failures == [
        "Aircraft numeric mismatch: 40.7639 != 40.7638",
    ]


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


def test_compare_json_value_accepts_wrapper_inf_strings():
    """Check JSON Inf strings compare with Python infinite values."""

    failures, compared = compare_json_value(
        {
            "positive": float("inf"),
            "negative": -float("inf"),
        },
        {
            "positive": "Inf",
            "negative": "-Inf",
        },
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


def test_compare_json_value_accepts_new_actual_history_fields():
    """Check current MATLAB history fields do not break old wrapper baselines."""

    failures, compared = compare_json_value(
        {
            "Mission": {
                "History": {
                    "SI": {
                        "Aero": {
                            "CL": [0.1],
                            "CD": [0.02],
                            "L_D": [5.0],
                            "Dwm": [0.0],
                        },
                        "Power": {
                            "DV": [123.0],
                            "Req": [456.0],
                        },
                    }
                }
            }
        },
        {
            "Mission": {
                "History": {
                    "SI": {
                        "Power": {
                            "Req": [456.0],
                        },
                    }
                }
            }
        },
    )

    assert compared > 0
    assert failures == []


def test_compare_json_value_reports_missing_current_history_fields():
    """Check missing current MATLAB history fields are still visible."""

    failures, compared = compare_json_value(
        {
            "Mission": {
                "History": {
                    "SI": {
                        "Power": {
                            "Req": [456.0],
                        },
                    }
                }
            }
        },
        {
            "Mission": {
                "History": {
                    "SI": {
                        "Aero": {
                            "CL": [0.1],
                        },
                        "Power": {
                            "DV": [123.0],
                            "Req": [456.0],
                        },
                    }
                }
            }
        },
    )

    assert compared > 0
    assert "Aircraft.Mission.History.SI.Aero missing from actual output" in failures
    assert "Aircraft.Mission.History.SI.Power.DV missing from actual output" in failures


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


def test_compare_json_value_ignores_prop_arch_parallel_connections():
    """Check Python zero-based ParConns do not obscure OutputAircraft parity."""

    failures, compared = compare_json_value(
        {
            "Specs": {
                "Propulsion": {
                    "PropArch": {
                        "ParConns": [[4], [5]],
                        "TrnType": [1, 1, 0, 0, 2, 2],
                    }
                }
            }
        },
        {
            "Specs": {
                "Propulsion": {
                    "PropArch": {
                        "ParConns": [[5.0], [6.0]],
                        "TrnType": [1, 1, 0, 0, 2, 2],
                    }
                }
            }
        },
    )

    assert compared > 0
    assert failures == []
