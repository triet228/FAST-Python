# src/fast_python/units.py

"""Unit conversions ported from FAST UnitConversionPkg."""


MASS_FACTORS = {
    "lbm": {"lbm": 1, "kg": 0.453592370000000, "slug": 0.0310809501715673},
    "kg": {"lbm": 2.20462262184878, "kg": 1, "slug": 0.0685217658567918},
    "slug": {"lbm": 32.1740485564304, "kg": 14.5939029372064, "slug": 1},
}

LENGTH_FACTORS = {
    "ft": {"ft": 1, "m": 0.3048, "km": 0.0003048, "mi": 0.000189393939393939, "naut mi": 0.000164578833693305},
    "m": {"ft": 3.28083989501312, "m": 1, "km": 0.001, "mi": 0.000621371192237334, "naut mi": 0.000539956803455724},
    "km": {"ft": 3280.83989501312, "m": 1000, "km": 1, "mi": 0.621371192237334, "naut mi": 0.539956803455724},
    "mi": {"ft": 5280, "m": 1609.344, "km": 1.609344, "mi": 1, "naut mi": 0.868976241900648},
    "naut mi": {"ft": 6076.11548556430, "m": 1852, "km": 1.852, "mi": 1.15077944802354, "naut mi": 1},
}

VELOCITY_FACTORS = {
    "ft/s": {"ft/s": 1, "m/s": 0.3048, "km/s": 0.0003048, "km/h": 1.09728, "mph": 0.681818181818182, "kts": 0.592483801295896, "ft/min": 60},
    "m/s": {"ft/s": 3.28083989501312, "m/s": 1, "km/s": 0.001, "km/h": 3.6, "mph": 2.23693629205440, "kts": 1.94384449244060, "ft/min": 196.850393700787},
    "km/s": {"ft/s": 3280.83989501312, "m/s": 1000, "km/s": 1, "km/h": 3600, "mph": 2236.93629205440, "kts": 1943.84449244060, "ft/min": 196850.393700787},
    "km/h": {"ft/s": 0.911344415281423, "m/s": 0.277777777777778, "km/s": 0.000277777777777778, "km/h": 1, "mph": 0.621371192237334, "kts": 0.539956803455724, "ft/min": 54.6806649168854},
    "mph": {"ft/s": 1.46666666666667, "m/s": 0.44704, "km/s": 0.00044704, "km/h": 1.609344, "mph": 1, "kts": 0.868976241900648, "ft/min": 88},
    "kts": {"ft/s": 1.68780985710120, "m/s": 0.514444444444445, "km/s": 0.000514444444444444, "km/h": 1.852, "mph": 1.15077944802354, "kts": 1, "ft/min": 101.268591426072},
    "ft/min": {"ft/s": 0.0166666666666667, "m/s": 0.00508, "km/s": 0.00000508, "km/h": 0.018288, "mph": 0.0113636363636364, "kts": 0.00987473002159827, "ft/min": 1},
}

FORCE_FACTORS = {
    "lbf": {"lbf": 1, "N": 4.44822161526050},
    "N": {"lbf": 0.224808943099711, "N": 1},
}

TSFC_FACTORS = {
    "SI": {
        "SI": 1,
        "Imp": 3600 / 0.224808943099711 * 2.204622621848776,
    },
    "Imp": {
        "SI": 1 / (3600 / 0.224808943099711 * 2.204622621848776),
        "Imp": 1,
    },
}


def convert_mass(value, oldunit, newunit):
    """Convert mass values between lbm, kg, and slug."""

    return convert_with_factors(value, oldunit, newunit, MASS_FACTORS, "mass")


def convert_length(value, oldunit, newunit):
    """Convert length values between FAST-supported length units."""

    return convert_with_factors(value, oldunit, newunit, LENGTH_FACTORS, "length")


def convert_velocity(value, oldunit, newunit):
    """Convert velocity values between FAST-supported velocity units."""

    return convert_with_factors(value, oldunit, newunit, VELOCITY_FACTORS, "velocity")


def convert_force(value, oldunit, newunit):
    """Convert force values between lbf and N."""

    return convert_with_factors(value, oldunit, newunit, FORCE_FACTORS, "force")


def convert_tsfc(value, oldunit, newunit):
    """Convert thrust-specific fuel consumption between SI and imperial units."""

    return convert_with_factors(value, oldunit, newunit, TSFC_FACTORS, "TSFC")


def convert_temperature(value, oldunit, newunit):
    """Convert temperature values between K, C, R, and F."""

    if oldunit not in ("K", "C", "R", "F") or newunit not in ("K", "C", "R", "F"):
        raise ValueError("Unsupported temperature unit. Use K, C, R, or F.")

    kelvin = apply_scalar(value, lambda item: temperature_to_kelvin(item, oldunit))
    return apply_scalar(kelvin, lambda item: temperature_from_kelvin(item, newunit))


def convert_with_factors(value, oldunit, newunit, factors, label):
    """Apply a multiplicative conversion table to scalars or nested lists."""

    if oldunit not in factors or newunit not in factors[oldunit]:
        units = ", ".join(sorted(factors))
        raise ValueError(f"Unsupported {label} unit. Supported units: {units}.")

    scale = factors[oldunit][newunit]
    return apply_scalar(value, lambda item: item * scale)


def apply_scalar(value, func):
    """Apply a scalar conversion recursively to tuples and lists."""

    if isinstance(value, list):
        return [apply_scalar(item, func) for item in value]

    if isinstance(value, tuple):
        return tuple(apply_scalar(item, func) for item in value)

    return func(value)


def temperature_to_kelvin(value, unit):
    """Return a scalar temperature in Kelvin."""

    if unit == "K":
        return value

    if unit == "C":
        return value + 273.15

    if unit == "R":
        return value / 1.8

    return (value - 32) * (5 / 9) + 273.15


def temperature_from_kelvin(value, unit):
    """Return a scalar Kelvin temperature in the requested unit."""

    if unit == "K":
        return value

    if unit == "C":
        return value - 273.15

    if unit == "R":
        return value * 1.8

    return value * 1.8 - 459.67


ConvMass = convert_mass
ConvLength = convert_length
ConvVel = convert_velocity
ConvForce = convert_force
ConvTSFC = convert_tsfc
ConvTemp = convert_temperature
