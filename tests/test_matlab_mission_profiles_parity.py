# tests/test_matlab_mission_profiles_parity.py

"""Optional MATLAB parity tests for every MissionProfilesPkg preset."""

import os
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from fast_python import profiles
from fast_python.compare import compare_json_value
from fast_python.io import build_json_data


def make_empty_aircraft():
    """Return the minimal aircraft accepted by constant profile presets."""

    return {}


def make_profile_aircraft():
    """Return compact performance inputs used by aircraft-dependent profiles."""

    return {
        "Specs": {
            "Performance": {
                "Range": 100000,
                "Alts": {
                    "Tko": 10,
                    "Crs": 3000,
                },
                "Vels": {
                    "Tko": 80,
                    "Crs": 0.6,
                },
            },
        },
    }


MISSION_PROFILE_CASES = [
    ("A320", profiles.A320, make_empty_aircraft, []),
    ("AEAProfile", profiles.AEAProfile, make_profile_aircraft, []),
    ("ATR42_600", profiles.ATR42_600, make_empty_aircraft, []),
    ("CeRAS", profiles.CeRAS, make_empty_aircraft, []),
    ("NotionalMission00", profiles.NotionalMission00, make_profile_aircraft, []),
    ("NotionalMission01", profiles.NotionalMission01, make_profile_aircraft, []),
    ("NotionalMission02", profiles.NotionalMission02, make_profile_aircraft, []),
    ("TakeoffTestProfile", profiles.TakeoffTestProfile, make_empty_aircraft, []),
    ("RegionalJetMission00", profiles.RegionalJetMission00, make_empty_aircraft, []),
    ("RegionalJetMission01", profiles.RegionalJetMission01, make_empty_aircraft, []),
    ("RegionalJetMission02", profiles.RegionalJetMission02, make_empty_aircraft, []),
    ("TurbopropMission00", profiles.TurbopropMission00, make_empty_aircraft, []),
    ("TurbopropMission01", profiles.TurbopropMission01, make_empty_aircraft, []),
    ("TurbopropMission02", profiles.TurbopropMission02, make_empty_aircraft, []),
    ("BRECruise00", profiles.BRECruise00, make_empty_aircraft, []),
    ("BRECruise01", profiles.BRECruise01, make_empty_aircraft, []),
    ("BRECruise02", profiles.BRECruise02, make_empty_aircraft, []),
    ("DiversionMission", profiles.DiversionMission, make_profile_aircraft, []),
    (
        "DiversionMission",
        lambda aircraft: profiles.DiversionMission(aircraft, 10),
        make_profile_aircraft,
        [10],
    ),
    ("ParametricRegional", profiles.ParametricRegional, make_profile_aircraft, []),
    ("LM100J_NoRsrv", profiles.LM100J_NoRsrv, make_profile_aircraft, []),
    ("ATRMissionBRE", profiles.ATRMissionBRE, make_empty_aircraft, []),
    ("ATRMissionEPASS", profiles.ATRMissionEPASS, make_empty_aircraft, []),
    ("LM100J", profiles.LM100J, make_profile_aircraft, []),
    ("ERJ", profiles.ERJ, make_profile_aircraft, []),
    ("ERJ_ClimbThenAccel", profiles.ERJ_ClimbThenAccel, make_profile_aircraft, []),
]


@pytest.fixture(scope="module")
def matlab_wrapper():
    """Start FAST-Python-Wrapper only when MATLAB parity is explicitly enabled."""

    if os.environ.get("FAST_PYTHON_RUN_MATLAB_PARITY") != "1":
        pytest.skip("Set FAST_PYTHON_RUN_MATLAB_PARITY=1 to run MATLAB parity tests.")

    wrapper_path = Path(
        os.environ.get(
            "FAST_PYTHON_WRAPPER_PATH",
            "C:/Users/homin/Projects/FAST-Python-Wrapper",
        )
    ).expanduser()
    fast_path = Path(
        os.environ.get(
            "FAST_PATH",
            os.environ.get("FAST_MATLAB_PATH", "C:/Users/homin/Projects/FAST"),
        )
    ).expanduser()

    if not wrapper_path.exists():
        pytest.skip(f"FAST-Python-Wrapper path not found: {wrapper_path}")

    if not fast_path.exists():
        pytest.skip(f"MATLAB FAST path not found: {fast_path}")

    if str(wrapper_path) not in sys.path:
        sys.path.insert(0, str(wrapper_path))

    wrapper_module = pytest.importorskip("wrapper")
    helper_module = pytest.importorskip("helper")
    wrapper = wrapper_module.FastWrapper(fast_path)
    wrapper.start()

    try:
        yield wrapper, helper_module.build_json_data
    finally:
        wrapper.stop()


@pytest.mark.parametrize(
    "matlab_name, python_applicator, aircraft_factory, matlab_args",
    MISSION_PROFILE_CASES,
)
def test_mission_profiles_pkg_matches_matlab(
    matlab_wrapper,
    matlab_name,
    python_applicator,
    aircraft_factory,
    matlab_args,
):
    """Compare one full MissionProfilesPkg preset against MATLAB FAST output."""

    wrapper, wrapper_build_json_data = matlab_wrapper
    aircraft = aircraft_factory()
    actual_aircraft = python_applicator(deepcopy(aircraft))
    actual = build_json_data(actual_aircraft["Mission"]["Profile"])
    expected = matlab_mission_profile(
        wrapper,
        wrapper_build_json_data,
        aircraft,
        matlab_name,
        matlab_args,
    )
    failures, compared = compare_json_value(actual, expected, "Mission.Profile")

    assert compared > 0

    if failures:
        preview = "\n".join(failures[:25])
        pytest.fail(f"{matlab_name} parity failures:\n{preview}")


def matlab_mission_profile(
    wrapper,
    wrapper_build_json_data,
    aircraft,
    matlab_name,
    matlab_args,
):
    """Return one MATLAB MissionProfilesPkg output as JSON-comparable data."""

    extra_args = "".join(
        f", {wrapper._to_matlab_literal(arg)}"
        for arg in matlab_args
    )
    try:
        wrapper.engine.evalc(
            f"""
            matlab_aircraft = {wrapper._to_matlab_literal(aircraft)};
            matlab_aircraft = MissionProfilesPkg.{matlab_name}(matlab_aircraft{extra_args});
            """,
            nargout=1,
        )
    except Exception as error:
        if matlab_name == "TakeoffTestProfile" and "convlength" in str(error):
            pytest.xfail(
                "MATLAB TakeoffTestProfile requires Aerospace Toolbox convlength."
            )

        raise

    matlab_aircraft = wrapper._to_python_data(wrapper.engine.workspace["matlab_aircraft"])
    return wrapper_build_json_data(matlab_aircraft["Mission"]["Profile"])
