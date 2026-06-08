# src/fast_python/cases.py

"""Convenience factories for native FAST aircraft and mission case inputs.

These helpers assemble aircraft and mission dictionaries from the Python
presets without reading JSON files. Every accessor returns a deep copy so tests
and exploratory runs can mutate the case data without contaminating future
factory calls.
"""

from copy import deepcopy

from fast_python.profiles import (
    mission_profile_a320,
    mission_profile_aea,
    mission_profile_atr42_600,
    mission_profile_ceras,
    mission_profile_erj,
    mission_profile_erj_climb_then_accel,
    mission_profile_lm100j,
    mission_profile_lm100j_no_reserve,
    mission_profile_notional_00,
    mission_profile_notional_01,
    mission_profile_notional_02,
    mission_profile_parametric_regional,
    mission_profile_regional_jet_00,
    mission_profile_regional_jet_01,
    mission_profile_regional_jet_02,
    mission_profile_turboprop_00,
    mission_profile_turboprop_01,
    mission_profile_turboprop_02,
)
from fast_python.specs import (
    aea,
    a320neo,
    aircraft_ceras,
    atr42,
    erj175lr,
    erj175lr_elec,
    erj190_e2,
    erj190_fe,
    example_aircraft,
    lm100j_conventional,
    lm100j_hybrid,
)


def mission_from_aircraft(profile_factory):
    """Return a mission factory that receives the paired aircraft."""

    def build_mission(aircraft):
        return profile_factory(aircraft)

    return build_mission


def constant_mission(profile_factory):
    """Return a mission factory for profiles that ignore aircraft specs."""

    def build_mission(_aircraft):
        return profile_factory()

    return build_mission


NATIVE_CASE_DEFINITIONS = [
    ("A320", a320neo, constant_mission(mission_profile_a320)),
    ("AEA", aea, mission_from_aircraft(mission_profile_aea)),
    ("ATR42", atr42, constant_mission(mission_profile_atr42_600)),
    ("CeRAS", aircraft_ceras, constant_mission(mission_profile_ceras)),
    ("ERJ175LR", erj175lr, mission_from_aircraft(mission_profile_erj)),
    (
        "ERJ175LR_ClimbThenAccel",
        erj175lr,
        mission_from_aircraft(mission_profile_erj_climb_then_accel),
    ),
    ("ERJ175LR_Elec", erj175lr_elec, mission_from_aircraft(mission_profile_erj)),
    ("ERJ190_E2", erj190_e2, mission_from_aircraft(mission_profile_erj)),
    ("ERJ190_FE", erj190_fe, mission_from_aircraft(mission_profile_erj)),
    ("Example_Notional00", example_aircraft, mission_from_aircraft(mission_profile_notional_00)),
    ("Example_Notional01", example_aircraft, mission_from_aircraft(mission_profile_notional_01)),
    ("Example_Notional02", example_aircraft, mission_from_aircraft(mission_profile_notional_02)),
    ("Example_RegionalJet00", example_aircraft, constant_mission(mission_profile_regional_jet_00)),
    ("Example_RegionalJet01", example_aircraft, constant_mission(mission_profile_regional_jet_01)),
    ("Example_RegionalJet02", example_aircraft, constant_mission(mission_profile_regional_jet_02)),
    ("Example_Turboprop00", example_aircraft, constant_mission(mission_profile_turboprop_00)),
    ("Example_Turboprop01", example_aircraft, constant_mission(mission_profile_turboprop_01)),
    ("Example_Turboprop02", example_aircraft, constant_mission(mission_profile_turboprop_02)),
    ("Example_ParametricRegional", example_aircraft, mission_from_aircraft(mission_profile_parametric_regional)),
    ("LM100J_Conventional", lm100j_conventional, mission_from_aircraft(mission_profile_lm100j_no_reserve)),
    ("LM100J_Hybrid", lm100j_hybrid, mission_from_aircraft(mission_profile_lm100j)),
]


def native_case(case_name):
    """Return native aircraft and mission dictionaries for a supported case."""

    key = case_name.lower()

    for name, aircraft_factory, mission_factory in NATIVE_CASE_DEFINITIONS:
        if key == name.lower():
            aircraft = aircraft_factory()
            return aircraft, mission_factory(aircraft)

    supported = ", ".join(native_case_names())
    raise ValueError(f"Unknown native FAST case {case_name!r}. Supported: {supported}.")


def native_case_names():
    """Return supported native aircraft/profile case names."""

    return [
        name
        for name, _aircraft_factory, _mission_factory in NATIVE_CASE_DEFINITIONS
    ]


def native_case_aircraft(case_name):
    """Return only the aircraft dictionary for a supported native case."""

    aircraft, _ = native_case(case_name)
    return deepcopy(aircraft)


def native_case_mission(case_name):
    """Return only the mission profile dictionary for a supported native case."""

    _, mission = native_case(case_name)
    return deepcopy(mission)
