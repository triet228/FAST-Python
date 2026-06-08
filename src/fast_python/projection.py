# src/fast_python/projection.py

"""Key performance parameter projections ported from ProjectionPkg."""

import math


class ProjectionError(ValueError):
    """Report unsupported aircraft classes or projection names."""


def kpp_projection(aircraft_class, year, kpp):
    """Project a FAST key performance parameter to a requested year.

    Inputs:
        aircraft_class: Aircraft class string, currently Turbofan or Turboprop.
        year: Entry-into-service year used by the projection curves.
        kpp: Name of the key performance parameter to project.

    Outputs:
        Scalar projected value in the same units as ProjectionPkg.KPPProjection.

    Assumptions:
        Coefficients mirror the MATLAB S-curve fits used by FAST for defaults
        such as battery specific energy and electric motor specific power.
    """

    aircraft_class = str(aircraft_class)
    kpp = str(kpp)
    year = float(year)

    if aircraft_class == "Turbofan":
        if kpp == "Cruise SFC":
            return sigmoid(year, 0.3335, 0.851, -0.0727, 1992)

        if kpp == "Total Takeoff T/ MTOW":
            return sigmoid(year, 0.15, 0.3465, 0.1132, 1980)

        if kpp == "OEW/MTOW":
            return 0.5123

        if kpp == "M(L/D)":
            return 14.44051

    elif aircraft_class == "Turboprop":
        if kpp == "Cruise SFC":
            return sigmoid(year, 0.3335, 0.6, -0.1012, 1993)

        if kpp == "Total Takeoff T/ MTOW":
            return sigmoid(year, 0.3, 0.9, 0.5334, 2011)

        if kpp == "OEW/MTOW":
            return 0.6284

        if kpp == "M(L/D)":
            return sigmoid(year, 1, 10, 0.1683, 2010)

    else:
        raise ProjectionError(
            "Current Class is not Supported. Please Enter: Piston, Turbofan, Turboprop."
        )

    if kpp == "Battery Specific Energy":
        return battery_specific_energy(year)

    if kpp == "Electric Motor Specific Power":
        return electric_motor_specific_power(year)

    raise ProjectionError(
        "Current KPP is not Supported. Please Enter: Cruise SFC, "
        "Total Takeoff T/ MTOW, OEW/MTOW, M(L/D), Battery Specific Energy, "
        "Electric Motor Specific Power."
    )


def sigmoid(year, low, high, growth_rate, inflection):
    """Evaluate the FAST logistic projection curve."""

    return low + ((high - low) / (1 + math.exp((-growth_rate) * (year - inflection))))


def battery_specific_energy(year):
    """Return projected battery specific energy in Wh/kg."""

    return 801.8 / (1 + math.exp(0.0607 * (2030.8 - year)))


def electric_motor_specific_power(year):
    """Return projected electric motor specific power in kW/kg."""

    return 37.8 / (1 + math.exp(-0.1213 * (year - 2030)))


KPPProjection = kpp_projection
