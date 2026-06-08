# src/fast_python/oew.py

"""Operating empty weight helpers ported from OEWPkg.

OEW helpers update aircraft weight estimates from regression data and propulsion
sizing outputs. Inputs and outputs are aircraft dictionaries in SI units; the
top-level iteration returns a deep copy so caller-owned sizing state is not
mutated unexpectedly.
"""

from copy import deepcopy

import numpy as np

from fast_python.propulsion import propulsion_sizing
from fast_python.regression import nlgpr, search_db


class OEWError(ValueError):
    """Report invalid OEW iteration inputs."""


def oew_iteration(aircraft):
    """Iterate airframe, propulsion, OEW, MTOW, and wing area estimates.

    Inputs:
        aircraft: Dictionary with processed aircraft specs, historical data,
            propulsion architecture, and initial component weights.

    Outputs:
        A deep-copied aircraft dictionary with updated Weight and Aero.S fields.

    Assumptions:
        This follows OEWPkg.OEWIteration. Turbofan airframe weight uses the
        precomputed Gaussian-process regression parameters from SpecProcessing;
        turboprop airframe weight uses FAST's linear airframe-vs-MTOW fit.
    """

    aircraft = deepcopy(aircraft)
    specs = aircraft["Specs"]
    weight = specs["Weight"]
    aclass = specs["TLAR"]["Class"]
    eis = specs["TLAR"]["EIS"]
    wing_loading = specs["Aero"]["W_S"]["SLS"]
    w_fuel = numeric_sum(weight.get("Fuel", 0))
    w_batt = numeric_sum(weight.get("Batt", 0))
    w_eg = numeric_sum(weight.get("EG", 0))
    w_em = numeric_sum(weight.get("EM", 0))
    w_pax = numeric_sum(weight.get("Payload", 0))
    w_crew = numeric_sum(weight.get("Crew", 0))
    w_eng = numeric_sum(weight.get("Engines", 0))
    w_eap = numeric_sum(weight.get("EAP", 0))
    w_cab = numeric_sum(weight.get("Cables", 0))
    frame_factor = weight.get("WairfCF", 1)
    w_frame_new = weight["OEW"] - w_eng - w_em - w_eg - w_eap - w_cab

    if w_frame_new < 0:
        w_frame_new = 0.4 * weight["MTOW"]

    tolerance = aircraft["Settings"]["OEW"]["Tol"]
    max_iter = int(aircraft["Settings"]["OEW"]["MaxIter"])
    iteration = 0
    error = 1

    if aclass == "Turbofan":
        aircraft, w_frame_new, w_eng, w_em, w_eg, w_cab, mtow, wing_area, iteration = (
            iterate_turbofan_oew(
                aircraft,
                w_frame_new,
                w_fuel,
                w_batt,
                w_pax,
                w_crew,
                w_em,
                w_eg,
                w_eng,
                w_eap,
                w_cab,
                wing_loading,
                eis,
                frame_factor,
                tolerance,
                max_iter,
            )
        )
    elif aclass == "Turboprop":
        fit = turboprop_airframe_fit(aircraft["HistData"]["AC"])

        while error > tolerance and iteration < max_iter:
            specs = aircraft["Specs"]
            w_frame_old = w_frame_new
            mtow = (
                w_frame_old
                + w_fuel
                + w_batt
                + w_pax
                + w_crew
                + w_em
                + w_eg
                + w_eng
                + w_eap
                + w_cab
            )
            specs["Power"]["SLS"] = mtow * specs["Power"]["P_W"]["SLS"]
            wing_area = mtow / wing_loading
            aircraft = propulsion_sizing(aircraft)
            specs = aircraft["Specs"]
            weight = specs["Weight"]
            w_eng_new = numeric_sum(weight["Engines"])
            w_em_new = numeric_sum(weight["EM"])
            w_eg_new = numeric_sum(weight["EG"])
            w_cab_new = numeric_sum(weight["Cables"])
            mtow = (
                mtow
                + w_eng_new
                - w_eng
                + w_em_new
                - w_em
                + w_eg_new
                - w_eg
                + w_cab_new
                - w_cab
            )
            w_frame_new = float(np.polyval(fit, mtow)) * frame_factor
            error = abs(w_frame_old - w_frame_new) / max(abs(w_frame_old), 1.0e-12)
            iteration += 1
            w_eng = w_eng_new
            w_em = w_em_new
            w_eg = w_eg_new
            w_cab = w_cab_new
    else:
        raise OEWError("Aircraft class not supported by OEWIteration.")

    oew = w_frame_new + w_em + w_eg + w_eng + w_eap + w_cab
    weight = aircraft["Specs"]["Weight"]
    weight["Engines"] = w_eng
    weight["EM"] = w_em
    weight["EG"] = w_eg
    weight["Cables"] = w_cab
    weight["Airframe"] = w_frame_new
    weight["OEW"] = oew
    weight["MTOW"] = mtow
    aircraft["Specs"]["Aero"]["S"] = wing_area
    aircraft.setdefault("Settings", {})["OEWIterations"] = iteration
    return aircraft


def iterate_turbofan_oew(
    aircraft,
    w_frame_new,
    w_fuel,
    w_batt,
    w_pax,
    w_crew,
    w_em,
    w_eg,
    w_eng,
    w_eap,
    w_cab,
    wing_loading,
    eis,
    frame_factor,
    tolerance,
    max_iter,
):
    """Run the turbofan OEW regression loop."""

    gravity = 9.81
    iteration = 0
    error = 1
    mtow = aircraft["Specs"]["Weight"]["MTOW"]
    wing_area = aircraft["Specs"]["Aero"].get("S", mtow / wing_loading)

    while error > tolerance and iteration < max_iter:
        specs = aircraft["Specs"]
        w_frame_old = w_frame_new
        mtow = (
            w_frame_old
            + w_fuel
            + w_batt
            + w_pax
            + w_crew
            + w_em
            + w_eg
            + w_eng
            + w_eap
            + w_cab
        )
        thrust = mtow * specs["Propulsion"]["T_W"]["SLS"] * gravity
        wing_area = mtow / wing_loading
        specs["Propulsion"]["Thrust"]["SLS"] = thrust
        aircraft = propulsion_sizing(aircraft)
        specs = aircraft["Specs"]
        weight = specs["Weight"]
        w_eng_new = numeric_sum(weight["Engines"])
        w_em_new = numeric_sum(weight["EM"])
        w_eg_new = numeric_sum(weight["EG"])
        w_cab_new = numeric_sum(weight["Cables"])
        mtow = (
            mtow
            + w_eng_new
            - w_eng
            + w_em_new
            - w_em
            + w_eg_new
            - w_eg
            + w_cab_new
            - w_cab
        )
        target = [wing_area, thrust, eis, mtow]
        preprocessing = aircraft["RegressionParams"]["OEW"]
        w_frame_new = regression_airframe_weight(
            aircraft["HistData"]["AC"],
            target,
            preprocessing,
        ) * frame_factor
        error = abs(w_frame_old - w_frame_new) / max(abs(w_frame_old), 1.0e-12)
        iteration += 1
        w_eng = w_eng_new
        w_em = w_em_new
        w_eg = w_eg_new
        w_cab = w_cab_new

    return aircraft, w_frame_new, w_eng, w_em, w_eg, w_cab, mtow, wing_area, iteration


def regression_airframe_weight(data_ac, target, preprocessing):
    """Return turbofan airframe weight from the precomputed GPR."""

    io_space = [
        ["Specs", "Aero", "S"],
        ["Specs", "Propulsion", "Thrust", "SLS"],
        ["Specs", "TLAR", "EIS"],
        ["Specs", "Weight", "MTOW"],
        ["Specs", "Weight", "Airframe"],
    ]
    mean, _ = nlgpr(
        data_ac,
        io_space,
        target,
        preprocessing=preprocessing,
    )
    return float(np.asarray(mean).reshape(-1)[0])


def turboprop_airframe_fit(data_ac):
    """Return the FAST linear airframe-vs-MTOW fit for turboprops."""

    _, airframe_rows = search_db(data_ac, ["Specs", "Weight", "Airframe"])
    _, mtow_rows = search_db(data_ac, ["Specs", "Weight", "MTOW"])
    airframe = np.asarray([row[1] for row in airframe_rows], dtype=float)
    mtow = np.asarray([row[1] for row in mtow_rows], dtype=float)
    valid = (~np.isnan(airframe)) & (~np.isnan(mtow))
    return np.polyfit(mtow[valid], airframe[valid], 1)


def numeric_sum(value):
    """Return a scalar sum for scalar or vector values."""

    array = np.asarray(value, dtype=float)

    if array.size == 0:
        return 0.0

    return float(np.sum(array))


OEWIteration = oew_iteration
