# src/fast_python/atmosphere.py

"""Atmosphere and gravity utilities ported from MissionSegsPkg."""

import math

import numpy as np


def gravity(altitude):
    """Return gravitational acceleration at altitude in m/s^2.

    Inputs:
        altitude: Scalar, list, or tuple of geometric altitudes in meters.

    Outputs:
        Local gravitational acceleration with the same container shape.

    Assumptions:
        FAST uses a spherical-Earth correction from sea-level standard gravity;
        this helper preserves that lightweight model for mission energy terms.
    """

    grav_sl = 9.80665
    earth_rad = 6.371009e6
    return apply_scalar(
        altitude,
        lambda item: grav_sl * (earth_rad / (earth_rad + item)) ** 2,
    )


def standard_atmosphere(altitude):
    """Return temperature, pressure, and density for altitude in meters.

    Inputs:
        altitude: Scalar, list, tuple, or NumPy array of altitudes between 0
            and 100000 m.

    Outputs:
        A tuple of temperature in K, pressure in Pa, and density in kg/m^3
        with the input shape preserved.

    Assumptions:
        Constants and layer breakpoints mirror FAST MissionSegsPkg.StdAtm.
    """

    if isinstance(altitude, np.ndarray):
        return standard_atmosphere_array(altitude)

    if isinstance(altitude, list) or isinstance(altitude, tuple):
        values = [standard_atmosphere(item) for item in altitude]
        return (
            [item[0] for item in values],
            [item[1] for item in values],
            [item[2] for item in values],
        )

    alt = altitude

    if alt < 0 or alt > 100000:
        raise ValueError("Altitude must be between 0 and 100000 m.")

    g = 9.81
    gas_constant = 287

    if alt < 11000:
        temp = 288.15 - 0.0065 * alt
        pressure = 101300 * (temp / 288.15) ** (-g / (gas_constant * -0.0065))
    elif alt < 20000:
        temp = 216.65
        pressure = 2.2609e4 * math.exp(-g * (alt - 11000) / (gas_constant * temp))
    elif alt < 32000:
        temp = 216.65 + 0.0010 * (alt - 20000)
        pressure = 5.4731e3 * (temp / 216.65) ** (-g / (gas_constant * 0.0010))
    elif alt < 47000:
        temp = 228.65 + 0.0028 * (alt - 32000)
        pressure = 866.8940 * (temp / 228.65) ** (-g / (gas_constant * 0.0028))
    elif alt < 51000:
        temp = 270.65
        pressure = 110.6427 * math.exp(-g * (alt - 47000) / (gas_constant * temp))
    elif alt < 71000:
        temp = 270.65 - 0.0028 * (alt - 51000)
        pressure = 66.7260 * (temp / 270.65) ** (-g / (gas_constant * -0.0028))
    elif alt < 85000:
        temp = 214.65 - 0.0020 * (alt - 71000)
        pressure = 3.9401 * (temp / 214.65) ** (-g / (gas_constant * -0.0020))
    else:
        temp = 186.65
        pressure = 0.3615 * math.exp(-g * (alt - 85000) / (gas_constant * temp))

    density = pressure / (gas_constant * temp)
    return temp, pressure, density


def standard_atmosphere_array(altitude):
    """Return StdAtm values for a NumPy altitude array."""

    alt = np.asarray(altitude, dtype=float)

    if alt.ndim == 0:
        return standard_atmosphere(float(alt))

    if np.any(alt < 0) or np.any(alt > 100000):
        raise ValueError("Altitude must be between 0 and 100000 m.")

    g = 9.81
    gas_constant = 287
    temp = np.zeros_like(alt, dtype=float)
    pressure = np.zeros_like(alt, dtype=float)
    section01 = (alt >= 0) & (alt < 11000)
    section02 = (alt >= 11000) & (alt < 20000)
    section03 = (alt >= 20000) & (alt < 32000)
    section04 = (alt >= 32000) & (alt < 47000)
    section05 = (alt >= 47000) & (alt < 51000)
    section06 = (alt >= 51000) & (alt < 71000)
    section07 = (alt >= 71000) & (alt < 85000)
    section08 = (alt >= 85000) & (alt <= 100000)

    if np.any(section01):
        temp[section01] = 288.15 - 0.0065 * alt[section01]
        pressure[section01] = (
            101300
            * (temp[section01] / 288.15) ** (-g / (gas_constant * -0.0065))
        )

    if np.any(section02):
        temp[section02] = 216.65
        pressure[section02] = 2.2609e4 * np.exp(
            -g * (alt[section02] - 11000) / (gas_constant * temp[section02])
        )

    if np.any(section03):
        temp[section03] = 216.65 + 0.0010 * (alt[section03] - 20000)
        pressure[section03] = (
            5.4731e3
            * (temp[section03] / 216.65) ** (-g / (gas_constant * 0.0010))
        )

    if np.any(section04):
        temp[section04] = 228.65 + 0.0028 * (alt[section04] - 32000)
        pressure[section04] = (
            866.8940
            * (temp[section04] / 228.65) ** (-g / (gas_constant * 0.0028))
        )

    if np.any(section05):
        temp[section05] = 270.65
        pressure[section05] = 110.6427 * np.exp(
            -g * (alt[section05] - 47000) / (gas_constant * temp[section05])
        )

    if np.any(section06):
        temp[section06] = 270.65 - 0.0028 * (alt[section06] - 51000)
        pressure[section06] = (
            66.7260
            * (temp[section06] / 270.65) ** (-g / (gas_constant * -0.0028))
        )

    if np.any(section07):
        temp[section07] = 214.65 - 0.0020 * (alt[section07] - 71000)
        pressure[section07] = (
            3.9401
            * (temp[section07] / 214.65) ** (-g / (gas_constant * -0.0020))
        )

    if np.any(section08):
        temp[section08] = 186.65
        pressure[section08] = 0.3615 * np.exp(
            -g * (alt[section08] - 85000) / (gas_constant * temp[section08])
        )

    density = pressure / (gas_constant * temp)
    return temp, pressure, density


def apply_scalar(value, func):
    """Apply a scalar function recursively to tuples and lists.

    Inputs:
        value: Scalar, list, or tuple.
        func: Callable that accepts one scalar altitude-related value.

    Outputs:
        Result with list/tuple shape preserved. This keeps vector-style
        atmosphere calls compatible with FAST wrapper JSON data.
    """

    if isinstance(value, list):
        return [apply_scalar(item, func) for item in value]

    if isinstance(value, tuple):
        return tuple(apply_scalar(item, func) for item in value)

    return func(value)


Gravity = gravity
StdAtm = standard_atmosphere
