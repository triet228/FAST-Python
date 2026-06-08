# tests/test_safety.py

"""Tests for native SafetyPkg ports."""

import numpy as np
import pytest

from fast_python.safety import (
    SafetyError,
    component_database,
    compare_cols,
    enumerate_failures,
    failure_model,
    fault_tree_analysis,
    idempotent_law,
    law_of_absorption,
)


def assert_array_equal(actual, expected):
    """Assert arrays match exactly."""

    np.testing.assert_array_equal(actual, np.asarray(expected, dtype=int))


def test_boolean_helper_functions_match_safety_pkg_examples():
    """Check SafetyPkg Boolean simplification helpers."""

    assert_array_equal(compare_cols([1, 2, 3], [1, 4, 3]), [0, 4, 0])
    assert_array_equal(
        enumerate_failures([[[1], [2]], [[3], [4]]]),
        [
            [1, 3],
            [1, 4],
            [2, 3],
            [2, 4],
        ],
    )
    assert_array_equal(
        idempotent_law(
            [
                [1, 1, 2],
                [3, 4, 3],
            ]
        ),
        [
            [1, 2],
            [3, 4],
        ],
    )
    assert_array_equal(
        law_of_absorption(
            [
                [1, 0],
                [1, 2],
                [3, 4],
            ]
        ),
        [
            [1, 0],
            [3, 4],
        ],
    )


def test_fault_tree_analysis_or_gate_matches_matlab():
    """Check a simple OR-gate fault tree against MATLAB output."""

    arch = [
        [0, 1, 1, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 1],
        [0, 0, 0, 0],
    ]
    probability, modes = fault_tree_analysis(arch, make_components())

    assert abs(probability - 0.131) < 1.0e-12
    assert modes == [["A"], ["B"], ["C"], ["D"]]


def test_fault_tree_analysis_and_gate_matches_matlab():
    """Check a simple AND-gate fault tree against MATLAB output."""

    arch = [
        [0, 1, 1, 0],
        [0, 0, 0, 2],
        [0, 0, 0, 2],
        [0, 0, 0, 0],
    ]
    probability, modes = fault_tree_analysis(arch, make_components())

    assert abs(probability - 0.1012) < 1.0e-12
    assert modes == [["A", ""], ["B", "C"], ["D", ""]]


def test_failure_model_and_component_database():
    """Check auxiliary SafetyPkg helpers."""

    probabilities = failure_model([0.1, 0.2], 2)

    np.testing.assert_allclose(
        probabilities,
        [1 - np.exp(-0.2), 1 - np.exp(-0.4)],
        atol=1.0e-12,
        rtol=0,
    )
    assert component_database("Battery") == {
        "Name": "Battery",
        "FailRate": 93.1e-6,
    }

    with pytest.raises(SafetyError, match="component"):
        component_database("Unknown")


def make_components():
    """Return a small component table for fault-tree tests."""

    return {
        "Name": ["A", "B", "C", "D"],
        "FailMode": ["FailA", "FailB", "FailC", "FailD"],
        "FailRate": [0.1, 0.01, 0.02, 0.001],
    }
