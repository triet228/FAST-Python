# tests/test_matlab_battery_parity.py

"""Optional MATLAB parity tests for BatteryPkg helpers."""

import numpy as np

from fast_python.battery import charging


def test_charging_mixed_power_vector_matches_matlab(matlab_wrapper):
    """Compare Charging's mixed-sign Preq branch behavior against MATLAB."""

    wrapper, _ = matlab_wrapper
    aircraft = make_battery_model_aircraft()
    preq = [500, -300]
    time = [60, 120]
    actual = [
        np.asarray(item, dtype=float).reshape(-1)
        for item in charging(
            aircraft,
            preq,
            time,
            90,
            100,
            10,
        )
    ]
    wrapper.engine.evalc(
        f"""
        matlab_aircraft = {wrapper._to_matlab_literal(aircraft)};
        [matlab_voltage, matlab_current, matlab_pout, matlab_capacity, ...
            matlab_soc, matlab_c_rate] = BatteryPkg.Charging(...
            matlab_aircraft, [500; -300], [60; 120], 90, 100, 10);
        """,
        nargout=1,
    )
    expected = [
        np.asarray(
            wrapper._to_python_data(wrapper.engine.workspace[name]),
            dtype=float,
        ).reshape(-1)
        for name in (
            "matlab_voltage",
            "matlab_current",
            "matlab_pout",
            "matlab_capacity",
            "matlab_soc",
            "matlab_c_rate",
        )
    ]

    for actual_item, expected_item in zip(actual, expected):
        np.testing.assert_allclose(actual_item, expected_item)


def make_battery_model_aircraft():
    """Return representative Lithium-ion cell parameters for battery tests."""

    return {
        "Settings": {
            "Analysis": {
                "Type": 1,
            }
        },
        "Specs": {
            "Battery": {
                "MaxExtVolCell": 4.088,
                "IntResist": 0.01,
                "ExpVol": 0.6,
                "ExpCap": 3.0,
                "CapCell": 3.2,
                "Degradation": 0,
            }
        },
    }
