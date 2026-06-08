# tests/test_cost.py

"""Tests for native CostPkg ports."""

import pytest

from fast_python.cost import CostError, BattRepCost, battery_replacement_cost


def assert_close(actual, expected, tolerance=1.0e-7):
    """Assert two numeric values are close."""

    assert abs(actual - expected) <= tolerance


def test_battery_replacement_cost_nmc_with_bms():
    """Check NMC battery replacement cost with BMS fraction included."""

    aircraft = make_cost_aircraft(1)

    cost = battery_replacement_cost(aircraft, 2026, 1, 5)

    assert_close(cost, 16523.42898821058)
    assert_close(BattRepCost(aircraft, 2026, 1, 5), cost)


def test_battery_replacement_cost_nmc_without_bms():
    """Check NMC battery replacement cost without BMS fraction."""

    cost = battery_replacement_cost(make_cost_aircraft(1), 2030, 0, 2)

    assert_close(cost, 16752.81197871884)


def test_battery_replacement_cost_lfp_with_bms():
    """Check LFP battery replacement cost with BMS fraction included."""

    cost = battery_replacement_cost(make_cost_aircraft(2), 2026, 1, 4)

    assert_close(cost, 15348.936091214617)


def test_battery_replacement_cost_lfp_without_bms():
    """Check LFP battery replacement cost without BMS fraction."""

    cost = battery_replacement_cost(make_cost_aircraft(2), 2035, 0, 3)

    assert_close(cost, 12508.948705178012)


def test_battery_replacement_cost_rejects_invalid_options():
    """Check invalid chemistry and BMS flags raise CostError."""

    with pytest.raises(CostError, match="ChemType"):
        battery_replacement_cost(make_cost_aircraft(3), 2026, 1, 5)

    with pytest.raises(CostError, match="BMS"):
        battery_replacement_cost(make_cost_aircraft(1), 2026, 2, 5)


def make_cost_aircraft(chemistry):
    """Return a minimal aircraft dictionary for BattRepCost."""

    return {
        "Specs": {
            "Battery": {
                "Chem": chemistry,
            },
            "Power": {
                "SpecEnergy": {
                    "Batt": 720000,
                },
            },
            "Weight": {
                "Batt": 1000,
            },
        }
    }
