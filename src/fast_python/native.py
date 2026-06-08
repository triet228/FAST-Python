# src/fast_python/native.py

"""Native FAST Python workflow orchestration."""

import warnings
from copy import deepcopy

from fast_python.aircraft import prepare_aircraft
from fast_python.analysis import eap_analysis
from fast_python.data_struct import is_missing, pre_spec_processing, spec_processing
from fast_python.mission import process_profile
from fast_python.propulsion import create_prop_arch, prop_arch_connections


class NativeRunError(RuntimeError):
    """Report unsupported native workflow inputs."""


def run_native(aircraft, mission=None, database=None):
    """Run the currently ported native FAST workflow.

    Inputs:
        aircraft: FAST aircraft dictionary.
        mission: Optional mission profile dictionary. If omitted,
            aircraft.Mission.Profile must already exist.
        database: Optional loaded IDEAS database for SpecProcessing.

    Outputs:
        A result dictionary with status, mtow, aircraft, log, and backend keys.

    Assumptions:
        The native workflow covers cases supported by the ported numerical
        modules. It is intentionally separate from the reference fixture
        backend used for full wrapper parity while remaining MATLAB packages
        are being converted.
    """

    aircraft = prepare_aircraft(deepcopy(aircraft))

    if mission is not None:
        aircraft.setdefault("Mission", {})["Profile"] = native_profile_from_mission(mission)
    elif "Mission" not in aircraft or "Profile" not in aircraft["Mission"]:
        raise NativeRunError("A mission profile is required for native FAST runs.")

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


RunNative = run_native


def native_profile_from_mission(mission):
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
