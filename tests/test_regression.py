# tests/test_regression.py

"""Tests for native RegressionPkg ports."""

import math

import numpy as np

from fast_python.regression import (
    build_data,
    nlgpr,
    prior_calculation,
    reg_processing,
    search_db,
    square_exp_kernel,
    vary_user_inputs,
)


def assert_array_close(actual, expected, tolerance=1.0e-9):
    """Assert arrays match within tolerance."""

    np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=0)


def sample_database():
    """Return a small database shaped like FAST aircraft data."""

    return {
        "AC1": {
            "Specs": {
                "Performance": {"Range": 1000},
                "Weight": {"MTOW": 100, "Fuel": 10},
            }
        },
        "AC2": {
            "Specs": {
                "Performance": {"Range": 2000},
                "Weight": {"MTOW": 200, "Fuel": 20},
            }
        },
        "AC3": {
            "Specs": {
                "Performance": {"Range": math.nan},
                "Weight": {"MTOW": 300, "Fuel": 30},
            }
        },
    }


def test_search_db_returns_values_and_filtered_struct():
    """Check nested database search and optional filtering."""

    database = sample_database()
    filtered, rows = search_db(database, ["Specs", "Weight", "MTOW"])
    only_ac2, filtered_rows = search_db(
        database,
        ["Specs", "Weight", "MTOW"],
        200,
    )

    assert rows == [["AC1", 100], ["AC2", 200], ["AC3", 300]]
    assert list(filtered) == ["AC1", "AC2", "AC3"]
    assert filtered_rows == [["AC2", 200]]
    assert list(only_ac2) == ["AC2"]


def test_square_exp_kernel_matches_fast_formula():
    """Check squared exponential kernel values."""

    result = square_exp_kernel([[1, 2]], [[2, 4]], [[4, 9, 16]])
    expected = 16 * math.exp(-0.3 * (((1 - 2) ** 2 / 4) + ((2 - 4) ** 2 / 9)))

    assert_array_close(result, [expected])


def test_prior_and_build_data_remove_nan_rows():
    """Check data matrix construction and hyperparameter tuning."""

    database = sample_database()
    io_space = [
        ["Specs", "Performance", "Range"],
        ["Specs", "Weight", "MTOW"],
        ["Specs", "Weight", "Fuel"],
    ]
    matrix, hyperparams = build_data(database, io_space, [1, 2])

    assert prior_calculation(database, io_space) == 20
    assert_array_close(matrix, [[1000, 100, 10], [2000, 200, 20]])
    assert hyperparams.shape == (3,)
    assert hyperparams[0] > hyperparams[1]


def test_reg_processing_and_nlgpr_return_stable_shapes():
    """Check regression preprocessing and prediction outputs."""

    database = sample_database()
    io_space = [
        ["Specs", "Performance", "Range"],
        ["Specs", "Weight", "MTOW"],
        ["Specs", "Weight", "Fuel"],
    ]
    prior = np.asarray([20])
    data_matrix, hyperparams, inverse_term = reg_processing(
        database,
        io_space,
        prior,
        [1, 1],
    )
    mean, variance = nlgpr(
        database,
        io_space,
        [1500, 150],
        weights=[1, 1],
        prior=prior,
        preprocessing={
            "DataMatrix": data_matrix,
            "HyperParams": hyperparams,
            "InverseTerm": inverse_term,
        },
    )

    assert data_matrix.shape == (2, 3)
    assert inverse_term.shape == (2, 2)
    assert mean.shape == (1,)
    assert variance.shape == (1,)
    assert mean[0] > 10
    assert mean[0] < 20


def test_vary_user_inputs_classifies_nan_values():
    """Check known/unknown regression input classification."""

    aircraft = {
        "Specs": {
            "Performance": {
                "Vels": {"Tko": 70, "Crs": math.nan},
                "Alts": {"Crs": 10000},
            },
            "Weight": {"MTOW": 1000},
            "Aero": {"W_S": {"SLS": 500}, "L_D": {"Crs": math.nan}},
            "Propulsion": {"T_W": {"SLS": 0.3}, "Thrust": {"SLS": math.nan}},
        }
    }

    known, unknown = vary_user_inputs(aircraft, "Turbofan")

    assert ["Specs", "Performance", "Vels", "Tko"] in known["names"]
    assert ["Specs", "Performance", "Vels", "Crs"] in unknown
    assert ["Specs", "Aero", "L_D", "Crs"] in unknown
    assert ["Specs", "Weight", "Fuel"] in unknown
