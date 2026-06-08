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
)
from fast_python.specs import aea, a320neo, aircraft_ceras, atr42


def native_case(case_name):
    """Return native aircraft and mission dictionaries for a supported case."""

    key = case_name.lower()

    if key == "a320":
        aircraft = a320neo()
        return aircraft, mission_profile_a320()

    if key == "aea":
        aircraft = aea()
        return aircraft, mission_profile_aea(aircraft)

    if key == "atr42":
        aircraft = atr42()
        return aircraft, mission_profile_atr42_600()

    if key == "ceras":
        aircraft = aircraft_ceras()
        return aircraft, mission_profile_ceras()

    supported = ", ".join(native_case_names())
    raise ValueError(f"Unknown native FAST case {case_name!r}. Supported: {supported}.")


def native_case_names():
    """Return supported native baseline case names."""

    return ["A320", "AEA", "ATR42", "CeRAS"]


def native_case_aircraft(case_name):
    """Return only the aircraft dictionary for a supported native case."""

    aircraft, _ = native_case(case_name)
    return deepcopy(aircraft)


def native_case_mission(case_name):
    """Return only the mission profile dictionary for a supported native case."""

    _, mission = native_case(case_name)
    return deepcopy(mission)
