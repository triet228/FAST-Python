# src/fast_python/profiles.py

"""Mission profile presets ported from FAST MissionProfilesPkg.

Preset helpers return raw mission-profile dictionaries in SI units. They do not
run process_profile(); segment indexing is intentionally deferred until an
aircraft's Settings determine the number of control points per segment.
"""

from copy import deepcopy

import numpy as np

from fast_python.units import convert_length, convert_velocity


def apply_profile(aircraft, profile):
    """Return an aircraft copy with the supplied mission profile attached.

    Inputs:
        aircraft: Aircraft dictionary.
        profile: Mission profile dictionary.

    Outputs:
        Deep-copied aircraft dictionary with Mission.Profile set to profile.

    Side effects:
        None on the caller's aircraft object.
    """

    result = deepcopy(aircraft)
    result.setdefault("Mission", {})["Profile"] = profile
    return result


def mission_profile_a320():
    """Return MissionProfilesPkg.A320's mission profile dictionary."""

    ranges = convert_length([3400 / 3, 3400 / 3, 3400 / 3, 200], "naut mi", "m")
    return {
        "Target": {
            "Valu": ranges + [30],
            "Type": ["Dist", "Dist", "Dist", "Dist", "Time"],
        },
        "Segs": [
            "Takeoff",
            "Climb",
            "Climb",
            "Cruise",
            "Climb",
            "Cruise",
            "Climb",
            "Cruise",
            "Descent",
            "Climb",
            "Cruise",
            "Descent",
            "Cruise",
            "Descent",
            "Landing",
        ],
        "ID": [1, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5],
        "AltBeg": convert_length(
            [0, 0, 10000, 35000, 35000, 37000, 37000, 39000, 39000, 1500, 15000, 15000, 1500, 1500, 0],
            "ft",
            "m",
        ),
        "AltEnd": convert_length(
            [0, 10000, 35000, 35000, 37000, 37000, 39000, 39000, 1500, 15000, 15000, 1500, 1500, 0, 0],
            "ft",
            "m",
        ),
        "ClbRate": [np.nan] * 15,
        "VelBeg": [
            0,
            0.3,
            convert_velocity(250, "kts", "m/s"),
            0.78,
            0.78,
            0.78,
            0.78,
            0.78,
            0.78,
            0.3,
            0.3,
            0.3,
            0.3,
            0.3,
            0.3,
        ],
        "VelEnd": [
            0.3,
            convert_velocity(250, "kts", "m/s"),
            0.78,
            0.78,
            0.78,
            0.78,
            0.78,
            0.78,
            0.3,
            0.3,
            0.3,
            0.3,
            0.3,
            0.3,
            0,
        ],
        "TypeBeg": [
            "Mach",
            "Mach",
            "TAS",
            "Mach",
            "Mach",
            "Mach",
            "Mach",
            "Mach",
            "Mach",
            "Mach",
            "Mach",
            "Mach",
            "Mach",
            "Mach",
            "Mach",
        ],
        "TypeEnd": [
            "Mach",
            "TAS",
            "Mach",
            "Mach",
            "Mach",
            "Mach",
            "Mach",
            "Mach",
            "Mach",
            "Mach",
            "Mach",
            "Mach",
            "Mach",
            "Mach",
            "Mach",
        ],
    }


def mission_profile_aea(aircraft):
    """Return MissionProfilesPkg.AEAProfile's mission profile dictionary."""

    alt_tko = aircraft["Specs"]["Performance"]["Alts"]["Tko"]
    vel_tko = aircraft["Specs"]["Performance"]["Vels"]["Tko"]
    alt_crs = 7829
    vel_crs = 0.747
    return {
        "Target": {
            "Valu": [convert_length(500, "naut mi", "m")],
            "Type": ["Dist"],
        },
        "Segs": ["Takeoff", "Climb", "Cruise", "Descent"],
        "ID": [1, 1, 1, 1],
        "AltBeg": [alt_tko, alt_tko, alt_crs, alt_crs],
        "AltEnd": [alt_tko, alt_crs, alt_crs, alt_tko],
        "ClbRate": [np.nan, np.nan, np.nan, np.nan],
        "VelBeg": [0, vel_tko, vel_crs, vel_crs],
        "VelEnd": [vel_tko, vel_crs, vel_crs, 1.2 * vel_tko],
        "TypeBeg": ["TAS", "TAS", "Mach", "Mach"],
        "TypeEnd": ["TAS", "Mach", "Mach", "TAS"],
    }


def mission_profile_atr42_600():
    """Return MissionProfilesPkg.ATR42_600's mission profile dictionary."""

    return {
        "Target": {
            "Valu": [
                convert_length(703, "naut mi", "m"),
                convert_length(150, "naut mi", "m"),
                45,
            ],
            "Type": ["Dist", "Dist", "Time"],
        },
        "Segs": [
            "Takeoff",
            "Climb",
            "Cruise",
            "Descent",
            "Descent",
            "Climb",
            "Cruise",
            "Cruise",
            "Descent",
            "Descent",
            "Landing",
        ],
        "ID": [1, 1, 1, 1, 1, 2, 2, 3, 3, 3, 3],
        "AltBeg": convert_length(
            [0, 0, 25000, 25000, 3000, 1500, 15000, 15000, 15000, 3000, 1500],
            "ft",
            "m",
        ),
        "AltEnd": convert_length(
            [0, 25000, 25000, 3000, 1500, 15000, 15000, 15000, 3000, 1500, 0],
            "ft",
            "m",
        ),
        "ClbRate": [np.nan, np.nan, np.nan, -6.096, np.nan, np.nan, np.nan, np.nan, -6.096, np.nan, np.nan],
        "VelBeg": convert_velocity([0, 160, 200, 200, 200, 160, 200, 200, 200, 200, 160], "kts", "m/s"),
        "VelEnd": convert_velocity([160, 200, 200, 200, 160, 200, 200, 200, 200, 160, 0], "kts", "m/s"),
        "TypeBeg": ["TAS", "EAS", "EAS", "EAS", "EAS", "EAS", "EAS", "EAS", "EAS", "EAS", "EAS"],
        "TypeEnd": ["EAS", "EAS", "EAS", "EAS", "EAS", "EAS", "EAS", "EAS", "EAS", "EAS", "EAS"],
    }


def mission_profile_ceras():
    """Return MissionProfilesPkg.CeRAS's mission profile dictionary."""

    alt_crs33 = convert_length(33000, "ft", "m")
    alt_crs35 = convert_length(35000, "ft", "m")
    return {
        "Target": {
            "Valu": [
                convert_length(2500 * 0.35, "naut mi", "m"),
                convert_length(2500 * 0.65, "naut mi", "m"),
            ],
            "Type": ["Dist", "Dist"],
        },
        "Segs": ["Climb", "Cruise", "Climb", "Cruise", "Descent"],
        "ID": [1, 1, 2, 2, 2],
        "AltBeg": [0, alt_crs33, alt_crs33, alt_crs35, alt_crs35],
        "AltEnd": [alt_crs33, alt_crs33, alt_crs35, alt_crs35, 0],
        "VelBeg": [0.2, 0.78, 0.78, 0.78, 0.78],
        "VelEnd": [0.78, 0.78, 0.78, 0.78, 0.2],
        "TypeBeg": ["Mach", "Mach", "Mach", "Mach", "Mach"],
        "TypeEnd": ["Mach", "Mach", "Mach", "Mach", "Mach"],
        "ClbRate": [np.nan, np.nan, np.nan, np.nan, np.nan],
    }


def mission_profile_notional_00(aircraft):
    """Return MissionProfilesPkg.NotionalMission00's profile dictionary."""

    perf = aircraft["Specs"]["Performance"]
    alt_tko = perf["Alts"]["Tko"]
    alt_crs = perf["Alts"]["Crs"]
    vel_tko = perf["Vels"]["Tko"]
    vel_crs = perf["Vels"]["Crs"]
    return {
        "Target": {
            "Valu": [perf["Range"]],
            "Type": ["Dist"],
        },
        "Segs": ["Takeoff", "Climb", "Cruise", "Descent", "Landing"],
        "ID": [1, 1, 1, 1, 1],
        "AltBeg": [alt_tko, alt_tko, alt_crs, alt_crs, alt_tko],
        "AltEnd": [alt_tko, alt_crs, alt_crs, alt_tko, alt_tko],
        "ClbRate": [np.nan, np.nan, np.nan, np.nan, np.nan],
        "VelBeg": [0, vel_tko, vel_crs, vel_crs, 1.2 * vel_tko],
        "VelEnd": [vel_tko, vel_crs, vel_crs, 1.2 * vel_tko, 0],
        "TypeBeg": ["TAS", "TAS", "Mach", "Mach", "TAS"],
        "TypeEnd": ["TAS", "Mach", "Mach", "TAS", "TAS"],
    }


def mission_profile_notional_01(aircraft):
    """Return MissionProfilesPkg.NotionalMission01's profile dictionary."""

    return mission_profile_notional_reserve(aircraft, 100, "Dist")


def mission_profile_notional_02(aircraft):
    """Return MissionProfilesPkg.NotionalMission02's profile dictionary."""

    return mission_profile_notional_reserve(aircraft, 45, "Time")


def mission_profile_notional_reserve(aircraft, reserve_value, reserve_type):
    """Return the shared notional profile with a second reserve mission."""

    perf = aircraft["Specs"]["Performance"]
    alt_tko = perf["Alts"]["Tko"]
    alt_crs = perf["Alts"]["Crs"]
    vel_tko = perf["Vels"]["Tko"]
    vel_crs = perf["Vels"]["Crs"]
    return {
        "Target": {
            "Valu": [perf["Range"], reserve_value],
            "Type": ["Dist", reserve_type],
        },
        "Segs": [
            "Takeoff",
            "Climb",
            "Cruise",
            "Descent",
            "Climb",
            "Cruise",
            "Descent",
            "Landing",
        ],
        "ID": [1, 1, 1, 1, 2, 2, 2, 2],
        "AltBeg": [alt_tko, alt_tko, alt_crs, alt_crs, 0.1 * alt_crs, 0.2 * alt_crs, 0.2 * alt_crs, alt_tko],
        "AltEnd": [alt_tko, alt_crs, alt_crs, 0.1 * alt_crs, 0.2 * alt_crs, 0.2 * alt_crs, alt_tko, alt_tko],
        "ClbRate": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        "VelBeg": [0, vel_tko, vel_crs, vel_crs, 1.2 * vel_tko, 1.5 * vel_tko, 1.5 * vel_tko, 1.2 * vel_tko],
        "VelEnd": [vel_tko, vel_crs, vel_crs, 1.2 * vel_tko, 1.5 * vel_tko, 1.5 * vel_tko, 1.2 * vel_tko, 0],
        "TypeBeg": ["TAS", "TAS", "Mach", "Mach", "TAS", "TAS", "TAS", "TAS"],
        "TypeEnd": ["TAS", "Mach", "Mach", "TAS", "TAS", "TAS", "TAS", "TAS"],
    }


def mission_profile_takeoff_test():
    """Return MissionProfilesPkg.TakeoffTestProfile's profile dictionary."""

    return {
        "Target": {
            "Valu": [convert_length(1.7, "naut mi", "m")],
            "Type": ["Dist"],
        },
        "Segs": ["DetailedTakeoff", "Climb", "Cruise"],
        "ID": [1, 1, 1],
        "AltBeg": convert_length([0, 0, 50], "ft", "m"),
        "AltEnd": convert_length([0, 50, 50], "ft", "m"),
        "ClbRate": [np.nan, np.nan, np.nan],
        "VelBeg": convert_velocity([0, 140, 140], "kts", "m/s"),
        "VelEnd": convert_velocity([140, 140, 140], "kts", "m/s"),
        "TypeBeg": ["TAS", "TAS", "TAS"],
        "TypeEnd": ["TAS", "TAS", "TAS"],
    }


def mission_profile_regional_jet_00():
    """Return MissionProfilesPkg.RegionalJetMission00's profile dictionary."""

    return mission_profile_from_arrays(
        [convert_length(1650, "naut mi", "m")],
        ["Dist"],
        ["Takeoff", "Climb", "Cruise", "Descent", "Landing"],
        [1, 1, 1, 1, 1],
        convert_length([0, 0, 35000, 35000, 0], "ft", "m"),
        convert_length([0, 35000, 35000, 0, 0], "ft", "m"),
        convert_velocity([0, 140, 460, 460, 160], "kts", "m/s"),
        convert_velocity([140, 460, 460, 160, 0], "kts", "m/s"),
        ["TAS", "TAS", "TAS", "TAS", "TAS"],
        ["TAS", "TAS", "TAS", "TAS", "TAS"],
    )


def mission_profile_regional_jet_01():
    """Return MissionProfilesPkg.RegionalJetMission01's profile dictionary."""

    return mission_profile_regional_jet_reserve(
        convert_length([1650, 100], "naut mi", "m"),
        ["Dist", "Dist"],
    )


def mission_profile_regional_jet_02():
    """Return MissionProfilesPkg.RegionalJetMission02's profile dictionary."""

    return mission_profile_regional_jet_reserve(
        [convert_length(2150, "naut mi", "m"), 45],
        ["Dist", "Time"],
    )


def mission_profile_regional_jet_reserve(target_values, target_types):
    """Return the shared regional-jet reserve profile."""

    return mission_profile_from_arrays(
        target_values,
        target_types,
        ["Takeoff", "Climb", "Cruise", "Descent", "Climb", "Cruise", "Descent", "Landing"],
        [1, 1, 1, 1, 2, 2, 2, 2],
        convert_length([0, 0, 35000, 35000, 3000, 5000, 5000, 0], "ft", "m"),
        convert_length([0, 35000, 35000, 3000, 5000, 5000, 0, 0], "ft", "m"),
        convert_velocity([0, 140, 460, 460, 160, 240, 240, 160], "kts", "m/s"),
        convert_velocity([140, 460, 460, 160, 240, 240, 160, 0], "kts", "m/s"),
        ["TAS", "TAS", "TAS", "TAS", "TAS", "TAS", "TAS", "TAS"],
        ["TAS", "TAS", "TAS", "TAS", "TAS", "TAS", "TAS", "TAS"],
    )


def mission_profile_turboprop_00():
    """Return MissionProfilesPkg.TurbopropMission00's profile dictionary."""

    return mission_profile_from_arrays(
        [convert_length(703, "naut mi", "m")],
        ["Dist"],
        ["Takeoff", "Climb", "Cruise", "Descent", "Landing"],
        [1, 1, 1, 1, 1],
        convert_length([0, 0, 22000, 22000, 0], "ft", "m"),
        convert_length([0, 22000, 22000, 0, 0], "ft", "m"),
        convert_velocity([0, 100, 200, 200, 120], "kts", "m/s"),
        convert_velocity([100, 200, 200, 120, 0], "kts", "m/s"),
        ["TAS", "TAS", "EAS", "EAS", "TAS"],
        ["TAS", "EAS", "EAS", "TAS", "TAS"],
    )


def mission_profile_turboprop_01():
    """Return MissionProfilesPkg.TurbopropMission01's profile dictionary."""

    return mission_profile_turboprop_reserve(
        convert_length([703, 100], "naut mi", "m"),
        ["Dist", "Dist"],
    )


def mission_profile_turboprop_02():
    """Return MissionProfilesPkg.TurbopropMission02's profile dictionary."""

    return mission_profile_turboprop_reserve(
        [convert_length(703, "naut mi", "m"), 45],
        ["Dist", "Time"],
    )


def mission_profile_turboprop_reserve(target_values, target_types):
    """Return the shared turboprop reserve profile."""

    return mission_profile_from_arrays(
        target_values,
        target_types,
        ["Takeoff", "Climb", "Cruise", "Descent", "Climb", "Cruise", "Descent", "Landing"],
        [1, 1, 1, 1, 2, 2, 2, 2],
        convert_length([0, 0, 22000, 22000, 3000, 5000, 5000, 0], "ft", "m"),
        convert_length([0, 22000, 22000, 3000, 5000, 5000, 0, 0], "ft", "m"),
        convert_velocity([0, 100, 200, 200, 120, 140, 140, 120], "kts", "m/s"),
        convert_velocity([100, 200, 200, 120, 140, 140, 120, 0], "kts", "m/s"),
        ["TAS", "TAS", "EAS", "EAS", "TAS", "EAS", "EAS", "TAS"],
        ["TAS", "EAS", "EAS", "TAS", "EAS", "EAS", "TAS", "TAS"],
    )


def mission_profile_bre_cruise_00():
    """Return MissionProfilesPkg.BRECruise00's profile dictionary."""

    return mission_profile_bre_cruise(
        [convert_length(1650, "naut mi", "m")],
        ["Dist"],
        1,
    )


def mission_profile_bre_cruise_01():
    """Return MissionProfilesPkg.BRECruise01's profile dictionary."""

    return mission_profile_bre_cruise(
        convert_length([1650, 100], "naut mi", "m"),
        ["Dist", "Dist"],
        2,
    )


def mission_profile_bre_cruise_02():
    """Return MissionProfilesPkg.BRECruise02's profile dictionary."""

    return mission_profile_bre_cruise(
        [convert_length(1650, "naut mi", "m"), 45],
        ["Dist", "Time"],
        2,
    )


def mission_profile_bre_cruise(target_values, target_types, count):
    """Return a CruiseBRE-only profile with the requested target count."""

    return mission_profile_from_arrays(
        target_values,
        target_types,
        ["CruiseBRE"] * count,
        list(range(1, count + 1)),
        convert_length([35000] * count, "ft", "m"),
        convert_length([35000] * count, "ft", "m"),
        convert_velocity([460] * count, "kts", "m/s"),
        convert_velocity([460] * count, "kts", "m/s"),
        ["TAS"] * count,
        ["TAS"] * count,
    )


def mission_profile_diversion(aircraft, dv=0):
    """Return MissionProfilesPkg.DiversionMission's profile dictionary."""

    vtko = aircraft["Specs"]["Performance"]["Vels"]["Tko"]
    vtrn = vtko + convert_velocity(dv, "kts", "m/s")
    vcrs = convert_velocity(250, "kts", "m/s")
    int_alt = convert_length(3000, "ft", "m")
    div_alt = convert_length(10000, "ft", "m")
    return {
        "Target": {
            "Valu": [convert_length(200, "naut mi", "m")],
            "Type": ["Dist"],
        },
        "Segs": ["Takeoff", "Climb", "Climb", "Cruise", "Descent", "Landing"],
        "ID": [1, 1, 1, 1, 1, 1],
        "AltBeg": [0, 0, int_alt, div_alt, div_alt, 0],
        "AltEnd": [0, int_alt, div_alt, div_alt, 0, 0],
        "ClbRate": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        "VelBeg": [0, vtko, vtrn, vcrs, vcrs, 1.1 * vtko],
        "VelEnd": [vtko, vtrn, vcrs, vcrs, 1.1 * vtko, 0],
        "TypeBeg": ["TAS", "TAS", "TAS", "TAS", "TAS", "TAS"],
        "TypeEnd": ["TAS", "TAS", "TAS", "TAS", "TAS", "TAS"],
    }


def mission_profile_parametric_regional(aircraft):
    """Return MissionProfilesPkg.ParametricRegional's profile dictionary."""

    perf = aircraft["Specs"]["Performance"]
    alt_tko = perf["Alts"]["Tko"]
    alt_crs = perf["Alts"]["Crs"]
    alt_clb = convert_length(3000, "ft", "m")
    alt_clb_to_crs = alt_crs - convert_length(1000, "ft", "m")
    res_alt_beg = convert_length(1500, "ft", "m")
    res_alt_clb = convert_length(3000, "ft", "m")
    res_alt_crs = convert_length(10000, "ft", "m")
    res_alt_clb_to_crs = res_alt_crs - convert_length(1000, "ft", "m")
    vel_tko = perf["Vels"]["Tko"]
    vel_crs = perf["Vels"]["Crs"]
    vel_apr = 1.2 * vel_tko
    vel_clb = convert_velocity(200, "kts", "m/s")
    vel_des = convert_velocity(200, "kts", "m/s")
    res_vel_clb = convert_velocity(200, "kts", "m/s")
    res_vel_des = convert_velocity(200, "kts", "m/s")
    res_vel_crs = convert_velocity(250, "kts", "m/s")

    return {
        "Target": {
            "Valu": [perf["Range"], 45],
            "Type": ["Dist", "Time"],
        },
        "Segs": [
            "Takeoff",
            "Climb",
            "Climb",
            "Climb",
            "Cruise",
            "Descent",
            "Descent",
            "Descent",
            "Climb",
            "Climb",
            "Climb",
            "Cruise",
            "Descent",
            "Descent",
            "Descent",
            "Landing",
        ],
        "ID": [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2],
        "AltBeg": [
            alt_tko,
            alt_tko,
            alt_clb,
            alt_clb_to_crs,
            alt_crs,
            alt_crs,
            alt_clb_to_crs,
            alt_clb,
            res_alt_beg,
            res_alt_clb,
            res_alt_clb_to_crs,
            res_alt_crs,
            res_alt_crs,
            res_alt_clb_to_crs,
            res_alt_clb,
            alt_tko,
        ],
        "AltEnd": [
            alt_tko,
            alt_clb,
            alt_clb_to_crs,
            alt_crs,
            alt_crs,
            alt_clb_to_crs,
            alt_clb,
            res_alt_beg,
            res_alt_clb,
            res_alt_clb_to_crs,
            res_alt_crs,
            res_alt_crs,
            res_alt_clb_to_crs,
            res_alt_clb,
            alt_tko,
            alt_tko,
        ],
        "VelBeg": [
            0,
            vel_tko,
            vel_clb,
            vel_clb,
            vel_crs,
            vel_crs,
            vel_des,
            vel_des,
            vel_apr,
            res_vel_clb,
            res_vel_clb,
            res_vel_crs,
            res_vel_crs,
            res_vel_des,
            res_vel_des,
            vel_apr,
        ],
        "VelEnd": [
            vel_tko,
            vel_clb,
            vel_clb,
            vel_crs,
            vel_crs,
            vel_des,
            vel_des,
            vel_apr,
            res_vel_clb,
            res_vel_clb,
            res_vel_crs,
            res_vel_crs,
            res_vel_des,
            res_vel_des,
            vel_apr,
            0,
        ],
        "TypeBeg": [
            "TAS",
            "TAS",
            "EAS",
            "EAS",
            "Mach",
            "Mach",
            "EAS",
            "EAS",
            "TAS",
            "EAS",
            "EAS",
            "TAS",
            "TAS",
            "EAS",
            "EAS",
            "TAS",
        ],
        "TypeEnd": [
            "TAS",
            "EAS",
            "EAS",
            "Mach",
            "Mach",
            "EAS",
            "EAS",
            "TAS",
            "EAS",
            "EAS",
            "TAS",
            "TAS",
            "EAS",
            "EAS",
            "TAS",
            "TAS",
        ],
        "ClbRate": [np.nan] * 16,
    }


def mission_profile_lm100j_no_reserve(aircraft):
    """Return MissionProfilesPkg.LM100J_NoRsrv's profile dictionary."""

    vel_beg = convert_velocity([0, 200, 210, 300, 0.59, 280, 160], "kts", "m/s")
    vel_end = convert_velocity([200, 200, 300, 300, 0.59, 160, 0], "kts", "m/s")
    vel_beg[4] = 0.59
    vel_end[4] = 0.59
    return {
        "Target": {
            "Valu": [aircraft["Specs"]["Performance"]["Range"]],
            "Type": ["Dist"],
        },
        "Segs": ["Takeoff", "Climb", "Climb", "Climb", "Cruise", "Descent", "Landing"],
        "ID": [1, 1, 1, 1, 1, 1, 1],
        "AltBeg": convert_length([0, 0, 10000, 17000, 25000, 25000, 0], "ft", "m"),
        "AltEnd": convert_length([0, 10000, 17000, 25000, 25000, 0, 0], "ft", "m"),
        "ClbRate": convert_velocity([np.nan, np.nan, 2000, 1500, np.nan, -1500, np.nan], "ft/min", "m/s"),
        "VelBeg": vel_beg,
        "VelEnd": vel_end,
        "TypeBeg": ["TAS", "TAS", "TAS", "TAS", "Mach", "TAS", "TAS"],
        "TypeEnd": ["TAS", "TAS", "TAS", "TAS", "Mach", "TAS", "TAS"],
        "PowerOpt": [1, 0, 0, 0, 0, 0, 0],
    }


def mission_profile_atr_mission_bre():
    """Return MissionProfilesPkg.ATRMissionBRE's profile dictionary."""

    return mission_profile_from_arrays(
        [convert_length(801, "naut mi", "m"), 45, convert_length(87, "naut mi", "m")],
        ["Dist", "Time", "Dist"],
        ["CruiseBRE", "CruiseBRE", "CruiseBRE"],
        [1, 2, 3],
        convert_length([25000, 25000, 25000], "ft", "m"),
        convert_length([25000, 25000, 25000], "ft", "m"),
        convert_velocity([300, 300, 300], "kts", "m/s"),
        convert_velocity([300, 300, 300], "kts", "m/s"),
        ["TAS", "TAS", "TAS"],
        ["TAS", "TAS", "TAS"],
    )


def mission_profile_atr_mission_epass():
    """Return MissionProfilesPkg.ATRMissionEPASS's profile dictionary."""

    return mission_profile_from_arrays(
        [convert_length(801, "naut mi", "m"), 45, convert_length(87, "naut mi", "m")],
        ["Dist", "Time", "Dist"],
        [
            "Takeoff",
            "Climb",
            "Climb",
            "Cruise",
            "Descent",
            "Descent",
            "Descent",
            "Climb",
            "Climb",
            "Cruise",
            "Cruise",
            "Descent",
            "Descent",
            "Landing",
        ],
        [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3],
        convert_length([0, 0, 24000, 25000, 25000, 24000, 3000, 1500, 9000, 10000, 10000, 10000, 3000, 0], "ft", "m"),
        convert_length([0, 24000, 25000, 25000, 24000, 3000, 1500, 9000, 10000, 10000, 10000, 3000, 0, 0], "ft", "m"),
        convert_velocity([0, 160, 160, 300, 300, 200, 200, 160, 160, 200, 200, 200, 200, 160], "kts", "m/s"),
        convert_velocity([160, 160, 300, 300, 200, 200, 160, 160, 200, 200, 200, 200, 160, 0], "kts", "m/s"),
        ["TAS", "EAS", "EAS", "TAS", "TAS", "EAS", "EAS", "EAS", "EAS", "EAS", "EAS", "EAS", "EAS", "EAS"],
        ["EAS", "EAS", "TAS", "TAS", "EAS", "EAS", "EAS", "EAS", "EAS", "EAS", "EAS", "EAS", "EAS", "TAS"],
    )


def mission_profile_lm100j(aircraft):
    """Return MissionProfilesPkg.LM100J's profile dictionary."""

    vel_beg = convert_velocity([0, 200, 210, 300, 0.59, 280, 210, 300, 300, 140], "kts", "m/s")
    vel_end = convert_velocity([200, 200, 300, 300, 0.59, 210, 300, 300, 140, 0], "kts", "m/s")
    vel_beg[4] = 0.59
    vel_end[4] = 0.59
    return {
        "Target": {
            "Valu": [aircraft["Specs"]["Performance"]["Range"], 45],
            "Type": ["Dist", "Time"],
        },
        "Segs": ["Takeoff", "Climb", "Climb", "Climb", "Cruise", "Descent", "Climb", "Cruise", "Descent", "Landing"],
        "ID": [1, 1, 1, 1, 1, 1, 2, 2, 2, 2],
        "AltBeg": convert_length([0, 0, 10000, 17000, 25000, 25000, 5000, 10000, 10000, 0], "ft", "m"),
        "AltEnd": convert_length([0, 10000, 17000, 25000, 25000, 5000, 10000, 10000, 0, 0], "ft", "m"),
        "ClbRate": convert_velocity([np.nan, np.nan, 2000, 1500, np.nan, -1500, 2000, np.nan, -1500, np.nan], "ft/min", "m/s"),
        "VelBeg": vel_beg,
        "VelEnd": vel_end,
        "TypeBeg": ["TAS", "TAS", "TAS", "TAS", "Mach", "TAS", "TAS", "TAS", "TAS", "TAS"],
        "TypeEnd": ["TAS", "TAS", "TAS", "TAS", "Mach", "TAS", "TAS", "TAS", "TAS", "TAS"],
        "PowerOpt": [1, 1, 1, 1, 1, 0, 1, 1, 0, 1],
    }


def mission_profile_erj(aircraft):
    """Return MissionProfilesPkg.ERJ's profile dictionary."""

    return mission_profile_erj_family(aircraft, climb_then_accel=False)


def mission_profile_erj_climb_then_accel(aircraft):
    """Return MissionProfilesPkg.ERJ_ClimbThenAccel's profile dictionary."""

    return mission_profile_erj_family(aircraft, climb_then_accel=True)


def mission_profile_erj_family(aircraft, climb_then_accel):
    """Return the shared ERJ regional profile family."""

    perf = aircraft["Specs"]["Performance"]
    alt_tko = perf["Alts"]["Tko"]
    alt_crs = perf["Alts"]["Crs"]
    alt_clb = convert_length(3000, "ft", "m")
    alt_clb_to_crs = alt_crs - convert_length(1000, "ft", "m")
    res_alt_beg = convert_length(1500, "ft", "m")
    res_alt_clb = convert_length(3000, "ft", "m")
    res_alt_crs = convert_length(10000, "ft", "m")
    res_alt_clb_to_crs = res_alt_crs - convert_length(1000, "ft", "m")
    vel_tko = perf["Vels"]["Tko"]
    vel_crs = perf["Vels"]["Crs"]
    vel_apr = 1.2 * vel_tko
    vel_clb = convert_velocity(200, "kts", "m/s")
    vel_des = convert_velocity(210 if climb_then_accel else 200, "kts", "m/s")
    res_vel_clb = convert_velocity(200, "kts", "m/s")
    res_vel_des = convert_velocity(200, "kts", "m/s")
    res_vel_crs = convert_velocity(250, "kts", "m/s")

    if climb_then_accel:
        alt_beg = [alt_tko, alt_tko, alt_clb, alt_crs, alt_crs, alt_crs, alt_crs, alt_clb]
        alt_end = [alt_tko, alt_clb, alt_crs, alt_crs, alt_crs, alt_crs, alt_clb, res_alt_beg]
    else:
        alt_beg = [alt_tko, alt_tko, alt_clb, alt_clb_to_crs, alt_crs, alt_crs, alt_clb_to_crs, alt_clb]
        alt_end = [alt_tko, alt_clb, alt_clb_to_crs, alt_crs, alt_crs, alt_clb_to_crs, alt_clb, res_alt_beg]

    return {
        "Target": {
            "Valu": [perf["Range"], convert_length(100, "naut mi", "m"), 45],
            "Type": ["Dist", "Dist", "Time"],
        },
        "Segs": [
            "Takeoff",
            "Climb",
            "Climb",
            "Climb",
            "Cruise",
            "Descent",
            "Descent",
            "Descent",
            "Climb",
            "Climb",
            "Climb",
            "Cruise",
            "Cruise",
            "Descent",
            "Descent",
            "Descent",
            "Landing",
        ],
        "ID": [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3],
        "AltBeg": alt_beg + [
            res_alt_beg,
            res_alt_clb,
            res_alt_clb_to_crs,
            res_alt_crs,
            res_alt_crs,
            res_alt_crs,
            res_alt_clb_to_crs,
            res_alt_clb,
            alt_tko,
        ],
        "AltEnd": alt_end + [
            res_alt_clb,
            res_alt_clb_to_crs,
            res_alt_crs,
            res_alt_crs,
            res_alt_crs,
            res_alt_clb_to_crs,
            res_alt_clb,
            alt_tko,
            alt_tko,
        ],
        "VelBeg": [
            0,
            vel_tko,
            vel_clb,
            vel_clb,
            vel_crs,
            vel_crs,
            vel_des,
            vel_des,
            vel_apr,
            res_vel_clb,
            res_vel_clb,
            res_vel_crs,
            res_vel_crs,
            res_vel_crs,
            res_vel_des,
            res_vel_des,
            vel_apr,
        ],
        "VelEnd": [
            vel_tko,
            vel_clb,
            vel_clb,
            vel_crs,
            vel_crs,
            vel_des,
            vel_des,
            vel_apr,
            res_vel_clb,
            res_vel_clb,
            res_vel_crs,
            res_vel_crs,
            res_vel_crs,
            res_vel_des,
            res_vel_des,
            vel_apr,
            0,
        ],
        "TypeBeg": ["TAS", "TAS", "EAS", "EAS", "Mach", "Mach", "EAS", "EAS", "TAS", "EAS", "EAS", "TAS", "TAS", "TAS", "EAS", "EAS", "TAS"],
        "TypeEnd": ["TAS", "EAS", "EAS", "Mach", "Mach", "EAS", "EAS", "TAS", "EAS", "EAS", "TAS", "TAS", "TAS", "EAS", "EAS", "TAS", "TAS"],
        "ClbRate": [np.nan] * 17,
    }


def mission_profile_from_arrays(target_values, target_types, segs, ids, alt_beg, alt_end, vel_beg, vel_end, type_beg, type_end):
    """Build a profile dictionary from MATLAB-style arrays."""

    return {
        "Target": {
            "Valu": target_values,
            "Type": target_types,
        },
        "Segs": segs,
        "ID": ids,
        "AltBeg": alt_beg,
        "AltEnd": alt_end,
        "ClbRate": [np.nan] * len(segs),
        "VelBeg": vel_beg,
        "VelEnd": vel_end,
        "TypeBeg": type_beg,
        "TypeEnd": type_end,
    }


def a320(aircraft):
    """Attach the A320 mission profile to an aircraft dictionary."""

    return apply_profile(aircraft, mission_profile_a320())


def aea_profile(aircraft):
    """Attach the AEA mission profile to an aircraft dictionary."""

    return apply_profile(aircraft, mission_profile_aea(aircraft))


def atr42_600(aircraft):
    """Attach the ATR42-600 mission profile to an aircraft dictionary."""

    return apply_profile(aircraft, mission_profile_atr42_600())


def ceras(aircraft):
    """Attach the CeRAS mission profile to an aircraft dictionary."""

    return apply_profile(aircraft, mission_profile_ceras())


def notional_mission00(aircraft):
    """Attach the NotionalMission00 profile to an aircraft dictionary."""

    return apply_profile(aircraft, mission_profile_notional_00(aircraft))


def notional_mission01(aircraft):
    """Attach the NotionalMission01 profile to an aircraft dictionary."""

    return apply_profile(aircraft, mission_profile_notional_01(aircraft))


def notional_mission02(aircraft):
    """Attach the NotionalMission02 profile to an aircraft dictionary."""

    return apply_profile(aircraft, mission_profile_notional_02(aircraft))


def takeoff_test_profile(aircraft):
    """Attach the TakeoffTestProfile profile to an aircraft dictionary."""

    return apply_profile(aircraft, mission_profile_takeoff_test())


def regional_jet_mission00(aircraft):
    """Attach the RegionalJetMission00 profile to an aircraft dictionary."""

    return apply_profile(aircraft, mission_profile_regional_jet_00())


def regional_jet_mission01(aircraft):
    """Attach the RegionalJetMission01 profile to an aircraft dictionary."""

    return apply_profile(aircraft, mission_profile_regional_jet_01())


def regional_jet_mission02(aircraft):
    """Attach the RegionalJetMission02 profile to an aircraft dictionary."""

    return apply_profile(aircraft, mission_profile_regional_jet_02())


def turboprop_mission00(aircraft):
    """Attach the TurbopropMission00 profile to an aircraft dictionary."""

    return apply_profile(aircraft, mission_profile_turboprop_00())


def turboprop_mission01(aircraft):
    """Attach the TurbopropMission01 profile to an aircraft dictionary."""

    return apply_profile(aircraft, mission_profile_turboprop_01())


def turboprop_mission02(aircraft):
    """Attach the TurbopropMission02 profile to an aircraft dictionary."""

    return apply_profile(aircraft, mission_profile_turboprop_02())


def bre_cruise00(aircraft):
    """Attach the BRECruise00 profile to an aircraft dictionary."""

    return apply_profile(aircraft, mission_profile_bre_cruise_00())


def bre_cruise01(aircraft):
    """Attach the BRECruise01 profile to an aircraft dictionary."""

    return apply_profile(aircraft, mission_profile_bre_cruise_01())


def bre_cruise02(aircraft):
    """Attach the BRECruise02 profile to an aircraft dictionary."""

    return apply_profile(aircraft, mission_profile_bre_cruise_02())


def diversion_mission(aircraft, dv=0):
    """Attach the DiversionMission profile to an aircraft dictionary."""

    return apply_profile(aircraft, mission_profile_diversion(aircraft, dv))


def parametric_regional(aircraft):
    """Attach the ParametricRegional profile to an aircraft dictionary."""

    return apply_profile(aircraft, mission_profile_parametric_regional(aircraft))


def lm100j_no_reserve(aircraft):
    """Attach the LM100J_NoRsrv profile to an aircraft dictionary."""

    return apply_profile(aircraft, mission_profile_lm100j_no_reserve(aircraft))


def atr_mission_bre(aircraft):
    """Attach the ATRMissionBRE profile to an aircraft dictionary."""

    return apply_profile(aircraft, mission_profile_atr_mission_bre())


def atr_mission_epass(aircraft):
    """Attach the ATRMissionEPASS profile to an aircraft dictionary."""

    return apply_profile(aircraft, mission_profile_atr_mission_epass())


def lm100j(aircraft):
    """Attach the LM100J profile to an aircraft dictionary."""

    return apply_profile(aircraft, mission_profile_lm100j(aircraft))


def erj(aircraft):
    """Attach the ERJ profile to an aircraft dictionary."""

    return apply_profile(aircraft, mission_profile_erj(aircraft))


def erj_climb_then_accel(aircraft):
    """Attach the ERJ_ClimbThenAccel profile to an aircraft dictionary."""

    return apply_profile(aircraft, mission_profile_erj_climb_then_accel(aircraft))


A320 = a320
AEAProfile = aea_profile
ATR42_600 = atr42_600
CeRAS = ceras
NotionalMission00 = notional_mission00
NotionalMission01 = notional_mission01
NotionalMission02 = notional_mission02
TakeoffTestProfile = takeoff_test_profile
RegionalJetMission00 = regional_jet_mission00
RegionalJetMission01 = regional_jet_mission01
RegionalJetMission02 = regional_jet_mission02
TurbopropMission00 = turboprop_mission00
TurbopropMission01 = turboprop_mission01
TurbopropMission02 = turboprop_mission02
BRECruise00 = bre_cruise00
BRECruise01 = bre_cruise01
BRECruise02 = bre_cruise02
DiversionMission = diversion_mission
ParametricRegional = parametric_regional
LM100J_NoRsrv = lm100j_no_reserve
ATRMissionBRE = atr_mission_bre
ATRMissionEPASS = atr_mission_epass
LM100J = lm100j
ERJ = erj
ERJ_ClimbThenAccel = erj_climb_then_accel
