# src/fast_python/retrofit.py

"""RetrofitPkg helpers ported from FAST.

The current retrofit module exposes the default option structure used by FAST's
RetrofitPkg. Values are plain Python scalars and nested dictionaries so future
retrofit drivers can consume them without needing MATLAB structs.
"""


def example_options():
    """Return RetrofitPkg.ExampleOptions defaults."""

    return {
        "NumMotors": 2,
        "ThrustSplit": 0.20,
        "PayDecrease": 0.5,
        "BattSpecEnergy": 0.5,
        "PW_EM": 10,
        "SavingsType": "Fuel",
    }


ExampleOptions = example_options
