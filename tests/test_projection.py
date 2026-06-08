# tests/test_projection.py

"""Tests for native ProjectionPkg ports."""

import math

import pytest

from fast_python.projection import ProjectionError, kpp_projection


def assert_close(actual, expected, tolerance=1.0e-12):
    """Assert two numeric values are close."""

    assert abs(actual - expected) <= tolerance


def test_turbofan_projection_matches_fast_formula():
    """Check representative turbofan KPP projections."""

    year = 2021
    cruise_sfc = 0.3335 + ((0.851 - 0.3335) / (1 + math.exp(0.0727 * (year - 1992))))
    takeoff_tw = 0.15 + ((0.3465 - 0.15) / (1 + math.exp(-0.1132 * (year - 1980))))

    assert_close(kpp_projection("Turbofan", year, "Cruise SFC"), cruise_sfc)
    assert_close(kpp_projection("Turbofan", year, "Total Takeoff T/ MTOW"), takeoff_tw)
    assert_close(kpp_projection("Turbofan", year, "OEW/MTOW"), 0.5123)
    assert_close(kpp_projection("Turbofan", year, "M(L/D)"), 14.44051)


def test_turboprop_projection_matches_fast_formula():
    """Check representative turboprop KPP projections."""

    year = 2025
    cruise_sfc = 0.3335 + ((0.6 - 0.3335) / (1 + math.exp(0.1012 * (year - 1993))))
    mach_ld = 1 + ((10 - 1) / (1 + math.exp(-0.1683 * (year - 2010))))

    assert_close(kpp_projection("Turboprop", year, "Cruise SFC"), cruise_sfc)
    assert_close(kpp_projection("Turboprop", year, "M(L/D)"), mach_ld)
    assert_close(kpp_projection("Turboprop", year, "OEW/MTOW"), 0.6284)


def test_electric_technology_projections_are_shared_by_class():
    """Check shared battery and electric motor S-curves."""

    year = 2030
    battery = 801.8 / (1 + math.exp(0.0607 * (2030.8 - year)))
    motor = 37.8 / (1 + math.exp(-0.1213 * (year - 2030)))

    assert_close(kpp_projection("Turbofan", year, "Battery Specific Energy"), battery)
    assert_close(kpp_projection("Turboprop", year, "Electric Motor Specific Power"), motor)


def test_projection_rejects_unknown_inputs():
    """Check unsupported projection requests fail clearly."""

    with pytest.raises(ProjectionError):
        kpp_projection("Rocket", 2030, "Cruise SFC")

    with pytest.raises(ProjectionError):
        kpp_projection("Turbofan", 2030, "Cabin Coffee Rate")
