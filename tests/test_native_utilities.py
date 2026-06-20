# tests/test_native_utilities.py

"""Tests for native FAST utility ports."""

import math

import numpy as np

from fast_python.atmosphere import gravity, standard_atmosphere
from fast_python.mission import compute_flight_conditions, process_profile
from fast_python.units import (
    convert_force,
    convert_length,
    convert_mass,
    convert_temperature,
    convert_tsfc,
    convert_velocity,
)


def assert_close(actual, expected, tolerance=1e-9):
    """Assert two numeric values are close."""

    assert abs(actual - expected) <= tolerance


def test_unit_conversions_match_fast_tables():
    """Check representative conversions from UnitConversionPkg."""

    assert_close(convert_mass(1, "kg", "lbm"), 2.20462262184878)
    assert_close(convert_length(1, "m", "ft"), 3.28083989501312)
    assert_close(convert_velocity(1, "kts", "m/s"), 0.514444444444445)
    assert_close(convert_force(1, "lbf", "N"), 4.44822161526050)
    assert_close(convert_temperature(0, "C", "K"), 273.15)
    assert_close(convert_tsfc(1.826, "SI", "Imp"), 64464.99444, 1.0e-5)
    assert_close(convert_tsfc(1103.245, "Imp", "SI"), 0.03124991148, 1.0e-10)


def test_standard_atmosphere_and_gravity_at_sea_level():
    """Check sea-level atmosphere and gravity values."""

    temp, pressure, density = standard_atmosphere(0)

    assert_close(temp, 288.15)
    assert_close(pressure, 101300)
    assert_close(density, 101300 / (287 * 288.15))
    assert_close(gravity(0), 9.80665)


def test_standard_atmosphere_accepts_lists():
    """Check vector-style atmosphere calls return vector-style outputs."""

    temps, pressures, densities = standard_atmosphere([0, 11000])

    assert len(temps) == 2
    assert len(pressures) == 2
    assert len(densities) == 2
    assert_close(temps[0], 288.15)
    assert_close(temps[1], 216.65)


def test_standard_atmosphere_accepts_numpy_arrays():
    """Check StdAtm preserves NumPy array shape like MATLAB."""

    temps, pressures, densities = standard_atmosphere(np.asarray([[0, 11000]]))

    assert temps.shape == (1, 2)
    assert pressures.shape == (1, 2)
    assert densities.shape == (1, 2)
    assert_close(temps[0, 0], 288.15)
    assert_close(temps[0, 1], 216.65)


def test_process_profile_adds_segment_indices():
    """Check mission profile validation and segment endpoint generation."""

    aircraft = {
        "Settings": {
            "TkoPoints": 3,
            "ClbPoints": 4,
            "CrsPoints": 5,
            "DesPoints": 6,
        },
        "Mission": {
            "Profile": {
                "Segs": ["Takeoff", "Climb", "Cruise", "Landing"],
                "Target": {"Valu": [math.nan], "Type": ["Dist"]},
                "ID": [1, 1, 1, 1],
                "AltBeg": [0, 0, 1000, 0],
                "AltEnd": [0, 1000, 1000, 0],
                "VelBeg": [0, 50, 100, 80],
                "VelEnd": [50, 100, 100, 0],
                "TypeBeg": ["TAS", "TAS", "TAS", "TAS"],
                "TypeEnd": ["TAS", "TAS", "TAS", "TAS"],
                "ClbRate": [math.nan, 5, math.nan, math.nan],
            }
        },
    }

    result = process_profile(aircraft)
    profile = result["Mission"]["Profile"]

    assert profile["SegPts"] == [3, 4, 5, 2]
    assert profile["SegBeg"] == [1, 3, 6, 10]
    assert profile["SegEnd"] == [3, 6, 10, 11]


def test_compute_flight_conditions_tas_at_sea_level():
    """Check TAS input at sea level."""

    eas, tas, mach, temp, pressure, density, visc = compute_flight_conditions(
        0,
        0,
        "TAS",
        100,
    )

    assert_close(eas, 100)
    assert_close(tas, 100)
    assert_close(mach, 100 / math.sqrt(1.4 * 287 * 288.15))
    assert_close(temp, 288.15)
    assert_close(pressure, 101300)
    assert_close(density, 101300 / (287 * 288.15))
    assert_close(visc, 1.81e-5)


def test_compute_flight_conditions_vector_eas():
    """Check vector-style EAS input returns vector-style values."""

    eas, tas, mach, temp, pressure, density, visc = compute_flight_conditions(
        [0, 10000],
        0,
        "EAS",
        [100, 100],
    )

    assert len(eas) == 2
    assert len(tas) == 2
    assert_close(eas[0], 100)
    assert tas[1] > tas[0]
    assert mach[1] > mach[0]
    assert temp[1] < temp[0]
    assert pressure[1] < pressure[0]
    assert density[1] < density[0]
    assert visc[1] < visc[0]
