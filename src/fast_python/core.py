# src/fast_python/core.py

"""Public FAST Python native run API."""

import warnings
from copy import deepcopy

from fast_python.aircraft import prepare_aircraft
from fast_python.analysis import eap_analysis
from fast_python.data_struct import is_missing, pre_spec_processing, spec_processing
from fast_python.mission import process_profile
from fast_python.propulsion import create_prop_arch, prop_arch_connections


class UnsupportedCaseError(RuntimeError):
    """Report an input case outside the current Python backend coverage."""


class FastPython:
    """Run the native Python implementation of FAST's JSON-facing workflow.

    Inputs:
        database: Optional loaded IDEAS database for SpecProcessing.

    Outputs:
        run() returns a dictionary with status, mtow, aircraft, log, and native
        backend metadata matching the wrapper's high-level return contract.

    Assumptions:
        The workflow covers aircraft and mission inputs supported by the
        currently ported Python modules. MATLAB-wrapper comparisons should use
        FAST-Python-Wrapper or the explicit reference helpers, not this runner.
    """

    def __init__(self, database=None):
        self.database = database

    def run(self, aircraft, mission=None):
        """Run a supported FAST aircraft and mission case natively in Python.

        Inputs:
            aircraft: InputAircraft-style dictionary. Compact PropArch strings
                are normalized before execution.
            mission: Optional Mission.json profile dictionary. If omitted,
                aircraft.Mission.Profile must already exist.

        Outputs:
            Dictionary with status, mtow in kg, full OutputAircraft data, log,
            and backend name.

        Assumptions:
            Unsupported inputs should fail loudly so missing native algorithm
            coverage can be ported or checked against FAST-Python-Wrapper.
        """

        return run(aircraft, mission, self.database)


def run(aircraft, mission=None, database=None):
    """Run FAST with the native Python backend.

    Inputs:
        aircraft: FAST aircraft dictionary.
        mission: Optional FAST mission profile dictionary. If omitted,
            aircraft.Mission.Profile must already exist.
        database: Optional loaded IDEAS database for SpecProcessing.

    Outputs:
        The same result dictionary returned by FastPython.run().
    """

    aircraft = prepare_aircraft(deepcopy(aircraft))

    if mission is not None:
        aircraft.setdefault("Mission", {})["Profile"] = profile_from_mission(mission)
    elif "Mission" not in aircraft or "Profile" not in aircraft["Mission"]:
        raise UnsupportedCaseError("A mission profile is required for FAST Python runs.")

    aircraft = pre_spec_processing(aircraft)
    settings = aircraft["Settings"]

    if is_missing(settings["Analysis"]["Type"]):
        warnings.warn(
            "Analysis type not provided; assuming on-design analysis (+1).",
            RuntimeWarning,
            stacklevel=2,
        )
        settings["Analysis"]["Type"] = 1

    if settings["Analysis"]["Type"] > 0:
        aircraft = spec_processing(aircraft, database)

    aircraft = create_prop_arch(aircraft)
    aircraft = prop_arch_connections(aircraft)
    aircraft = process_profile(aircraft)
    aircraft = eap_analysis(aircraft)
    mtow = aircraft["Specs"]["Weight"]["MTOW"]

    return {
        "status": "success",
        "mtow": float(mtow),
        "aircraft": aircraft,
        "log": "FAST-Python native backend completed the ported workflow.",
        "backend": "native",
    }


def profile_from_mission(mission):
    """Return a mission profile from a raw profile or wrapper object.

    Inputs:
        mission: Either a raw MissionProfilesPkg-style dictionary or a wrapper
            object containing a Profile dictionary.

    Outputs:
        A deep copy of the profile dictionary used by process_profile().

    Assumptions:
        Wrapper JSON often nests the profile under "Profile", while native
        preset helpers return the profile directly. Accepting both keeps the
        Python backend usable from either entry point without mutating inputs.
    """

    profile = deepcopy(mission)

    if isinstance(profile, dict) and isinstance(profile.get("Profile"), dict):
        return deepcopy(profile["Profile"])

    return profile
