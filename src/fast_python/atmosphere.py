# src/fast_python/atmosphere.py

"""Atmosphere and gravity utilities ported from MissionSegsPkg."""

import math


def gravity(altitude):
    """Return gravitational acceleration at altitude in m/s^2."""

    grav_sl = 9.80665
    earth_rad = 6.371009e6
    return apply_scalar(
        altitude,
        lambda item: grav_sl * (earth_rad / (earth_rad + item)) ** 2,
    )


def standard_atmosphere(altitude):
    """Return temperature, pressure, and density for altitude in meters.

    Inputs:
        altitude: Scalar or list of altitudes between 0 and 100000 m.

    Outputs:
        A tuple of temperature in K, pressure in Pa, and density in kg/m^3.

    Assumptions:
        Constants and layer breakpoints mirror FAST MissionSegsPkg.StdAtm.
    """

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


def apply_scalar(value, func):
    """Apply a scalar function recursively to tuples and lists."""

    if isinstance(value, list):
        return [apply_scalar(item, func) for item in value]

    if isinstance(value, tuple):
        return tuple(apply_scalar(item, func) for item in value)

    return func(value)


Gravity = gravity
StdAtm = standard_atmosphere
