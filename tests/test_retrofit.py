# tests/test_retrofit.py

"""Tests for RetrofitPkg helper ports."""

from fast_python.retrofit import ExampleOptions, example_options


def test_example_options_matches_matlab_defaults():
    """Check RetrofitPkg.ExampleOptions default values."""

    expected = {
        "NumMotors": 2,
        "ThrustSplit": 0.20,
        "PayDecrease": 0.5,
        "BattSpecEnergy": 0.5,
        "PW_EM": 10,
        "SavingsType": "Fuel",
    }

    assert example_options() == expected
    assert ExampleOptions() == expected
