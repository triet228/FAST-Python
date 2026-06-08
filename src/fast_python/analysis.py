# src/fast_python/analysis.py

"""Aircraft analysis loop ported from FAST EAPAnalysis.

The public loop consumes already-processed aircraft dictionaries, flies the
mission, resizes energy sources, and iterates MTOW/OEW until the FAST
convergence criteria are satisfied. Inputs and outputs are SI-valued FAST
dictionaries; helper functions preserve scalar-versus-vector source weights so
single-source and multi-source aircraft serialize like wrapper output.
"""

from copy import deepcopy

import numpy as np

from fast_python.battery import resize_battery
from fast_python.data_struct import clear_mission, init_mission_history
from fast_python.mission import fly_mission
from fast_python.oew import oew_iteration
from fast_python.propulsion import as_vector, propulsion_sizing


class AnalysisError(ValueError):
    """Report invalid aircraft analysis inputs."""


def eap_analysis(aircraft, analysis_type=None, max_iter=None):
    """Run FAST's electric-aircraft-propulsion analysis loop.

    Inputs:
        aircraft: Dictionary after mission profile processing and propulsion
            architecture setup.
        analysis_type: Optional override for Settings.Analysis.Type.
        max_iter: Optional sizing-loop iteration cap.

    Outputs:
        A deep-copied aircraft dictionary with sized weights, wing area, and
        mission history populated.

    Assumptions:
        This ports EAPAnalysis for the native Python packages currently
        available. Detailed visualization, plotting, and battery degradation
        side effects remain outside this numerical loop.
    """

    aircraft = deepcopy(aircraft)
    settings = aircraft["Settings"]
    specs = aircraft["Specs"]
    prop_arch = specs["Propulsion"]["PropArch"]
    source_type = as_vector(prop_arch["SrcType"])
    fuel_sources = source_type == 1
    battery_sources = source_type == 0
    analysis_type = settings["Analysis"]["Type"] if analysis_type is None else analysis_type
    max_iter = settings["Analysis"]["MaxIter"] if max_iter is None else max_iter
    max_iter = int(max_iter)
    mtow = specs["Weight"]["MTOW"]
    wfuel = initial_source_weight(specs["Weight"].get("Fuel"), fuel_sources)
    wbatt = initial_source_weight(specs["Weight"].get("Batt"), battery_sources)
    wing_loading = specs["Aero"]["W_S"]["SLS"]

    if is_missing(mtow):
        raise AnalysisError("MTOW is NaN.")

    if is_missing(wing_loading):
        raise AnalysisError("Wing loading is NaN.")

    settings["DetailedBatt"] = detailed_battery_enabled(specs)
    specs["Aero"]["S"] = mtow / wing_loading
    settings["Converged"] = 1

    if analysis_type < 0:
        mtow = (
            specs["Weight"]["OEW"]
            + specs["Weight"]["Crew"]
            + specs["Weight"]["Payload"]
            + sum_weight(wfuel)
            + sum_weight(wbatt)
        )
        specs["Weight"]["MTOW"] = mtow

    aircraft = init_mission_history(aircraft)
    iteration = 0

    while iteration < max_iter:
        if iteration > 0:
            if analysis_type > 0:
                aircraft = oew_iteration(aircraft)

            aircraft = clear_mission(aircraft)
        else:
            aircraft = propulsion_sizing(aircraft)

        mtow = aircraft["Specs"]["Weight"]["MTOW"]
        aircraft = fly_mission(aircraft)
        history = aircraft["Mission"]["History"]["SI"]
        fburn = history["Weight"]["Fburn"][-1]
        dwfuel = np.asarray(fburn, dtype=float) - wfuel

        if analysis_type == -2:
            dwbatt = np.zeros_like(wbatt)
        else:
            aircraft = resize_battery(aircraft)
            dwbatt = initial_source_weight(
                aircraft["Specs"]["Weight"].get("Batt"),
                battery_sources,
            ) - wbatt

        mtow_new = mtow + float(np.sum(dwfuel)) + float(np.sum(dwbatt))
        dmtow = mtow_new - mtow
        fuel_conv = convergence_error(dwfuel, wfuel)
        batt_conv = convergence_error(dwbatt, wbatt)
        mtow_conv = abs(dmtow) / mtow
        wfuel = wfuel + dwfuel
        wbatt = wbatt + dwbatt
        weight = aircraft["Specs"]["Weight"]
        weight["MTOW"] = mtow_new
        weight["Fuel"] = restore_source_weight(wfuel)
        weight["Batt"] = restore_source_weight(wbatt)

        if analysis_type > -2:
            weight["OEW"] = (
                mtow_new
                - sum_weight(wfuel)
                - sum_weight(wbatt)
                - weight["Payload"]
                - weight["Crew"]
            )

        iteration += 1

        if iteration == 1 and analysis_type > 0:
            continue

        if (
            not np.any(fuel_conv > 1.0e-3)
            and not np.any(batt_conv > 1.0e-3)
            and not mtow_conv > 1.0e-3
        ):
            break

    if iteration == max_iter and analysis_type > 0:
        aircraft["Settings"]["Converged"] = 0

    return aircraft


def initial_source_weight(value, source_mask):
    """Return fuel or battery source weights as a numeric vector."""

    count = max(1, int(np.sum(source_mask)))

    if is_missing(value):
        return np.zeros(count)

    array = np.asarray(value, dtype=float).reshape(-1)

    if array.size == 1 and count > 1:
        return np.ones(count) * array[0]

    return array


def restore_source_weight(value):
    """Return a scalar for one source weight, otherwise a list."""

    array = np.asarray(value, dtype=float).reshape(-1)

    if array.size == 1:
        return float(array[0])

    return array.tolist()


def convergence_error(delta, baseline):
    """Return MATLAB-style relative convergence, treating NaN as zero."""

    with np.errstate(divide="ignore", invalid="ignore"):
        conv = np.abs(delta) / baseline

    conv[np.isnan(conv)] = 0
    return conv


def detailed_battery_enabled(specs):
    """Return 1 when detailed battery cell counts are prescribed."""

    battery = specs["Power"]["Battery"]

    if is_missing(battery.get("SerCells")) or is_missing(battery.get("ParCells")):
        return 0

    return 1


def sum_weight(value):
    """Return scalar sum for a scalar or source-weight vector."""

    return float(np.sum(np.asarray(value, dtype=float)))


def is_missing(value):
    """Return True for FAST-style missing numeric values."""

    if value is None:
        return True

    try:
        return bool(np.isnan(value))
    except TypeError:
        return False


EAPAnalysis = eap_analysis
