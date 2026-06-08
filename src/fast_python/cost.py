# src/fast_python/cost.py

"""Cost helpers ported from FAST CostPkg."""

import numpy as np


class CostError(ValueError):
    """Report invalid cost-model inputs."""


def battery_replacement_cost(aircraft, year, bms, lifespan):
    """Return FAST's battery replacement cost estimate.

    Inputs:
        aircraft: Aircraft dictionary with Specs.Battery.Chem,
            Specs.Power.SpecEnergy.Batt, and Specs.Weight.Batt.
        year: Calendar year for the projected battery cell and BMS costs.
        bms: 1 to include battery-management-system cost, 0 to exclude it.
        lifespan: Battery-system lifespan in years for discounting.

    Outputs:
        Discounted battery replacement cost in dollars.

    Assumptions:
        This mirrors CostPkg.BattRepCost, including its chemistry-specific
        polynomial regressions and fixed 7 percent discount rate.
    """

    specs = aircraft["Specs"]
    chemistry = specs["Battery"]["Chem"]
    years = np.asarray([2023, 2026, 2030, 2035], dtype=float)
    bms_lambda = bms_cost_fraction(years, chemistry, bms, year)
    capacity_cost = battery_capacity_cost(years, chemistry, year)
    specific_energy = specs["Power"]["SpecEnergy"]["Batt"] / 3600 / 1000
    rated_capacity = specific_energy * specs["Weight"]["Batt"]
    discount_rate = 0.07
    return (1 + bms_lambda / 100) * capacity_cost * rated_capacity / (
        (1 + discount_rate) ** lifespan
    )


def bms_cost_fraction(years, chemistry, bms, year):
    """Return the BMS cost percentage from FAST's chemistry curves."""

    if bms == 0:
        return 0

    if bms != 1:
        raise CostError('Invalid BMS requirement. Use "1" for YES or "0" for NO.')

    if chemistry == 1:
        values = np.asarray([2, 2.2, 3, 3.5], dtype=float)
        return float(np.polyval(np.polyfit(years, values, 1), year))

    if chemistry == 2:
        values = np.asarray([2.2, 3.1, 3.5, 2.6], dtype=float)
        return float(np.polyval(np.polyfit(years, values, 2), year))

    raise CostError('Invalid ChemType input. Use "1" for NMC or "2" for LFP.')


def battery_capacity_cost(years, chemistry, year):
    """Return the unit capacity replacement cost in dollars per kWh."""

    if chemistry == 1:
        values = np.asarray([127.61, 112.42, 96.52, 76.35], dtype=float)
        return float(np.polyval(np.polyfit(years, values, 2), year))

    if chemistry == 2:
        values = np.asarray([120.91, 97.56, 82.73, 76.62], dtype=float)
        return float(np.polyval(np.polyfit(years, values, 3), year))

    raise CostError('Invalid ChemType input. Use "1" for NMC or "2" for LFP.')


BattRepCost = battery_replacement_cost
