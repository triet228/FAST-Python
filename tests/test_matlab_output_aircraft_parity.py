# tests/test_matlab_output_aircraft_parity.py

"""Optional end-to-end OutputAircraft parity tests against MATLAB FAST."""

import os
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from fast_python.cases import native_case, native_case_names
from fast_python.compare import compare_json_value
from fast_python.core import run
from fast_python.io import build_json_data


OUTPUT_PARITY_MATLAB_CASES = {
    "A320": ("A320Neo", "A320"),
    "AEA": ("AEA", "AEAProfile"),
    "ATR42": ("ATR42", "ATR42_600"),
    "CeRAS": ("CeRAS", "CeRAS"),
    "ERJ175LR": ("ERJ175LR", "ERJ"),
    "ERJ175LR_ClimbThenAccel": ("ERJ175LR", "ERJ_ClimbThenAccel"),
    "ERJ175LR_Elec": ("ERJ175LR_Elec", "ERJ"),
    "ERJ190_E2": ("ERJ190_E2", "ERJ"),
    "ERJ190_FE": ("ERJ190_FE", "ERJ"),
    "Example_Notional00": ("Example", "NotionalMission00"),
    "Example_Notional01": ("Example", "NotionalMission01"),
    "Example_Notional02": ("Example", "NotionalMission02"),
    "Example_RegionalJet00": ("Example", "RegionalJetMission00"),
    "Example_RegionalJet01": ("Example", "RegionalJetMission01"),
    "Example_RegionalJet02": ("Example", "RegionalJetMission02"),
    "Example_Turboprop00": ("Example", "TurbopropMission00"),
    "Example_Turboprop01": ("Example", "TurbopropMission01"),
    "Example_Turboprop02": ("Example", "TurbopropMission02"),
    "Example_ParametricRegional": ("Example", "ParametricRegional"),
    "LM100J_Conventional": ("LM100J_Conventional", "LM100J_NoRsrv"),
    "LM100J_Hybrid": ("LM100J_Hybrid", "LM100J"),
}
NONCONVERGED_OUTPUT_CASES = {
    "A320": (
        "MATLAB FAST now flies A320Neo with AerodynamicsPkg.DragPolar; "
        "FAST-Python still uses the constant-L/D mission model for native "
        "end-to-end sizing."
    ),
    "ERJ175LR": (
        "MATLAB FAST now flies ERJ175LR with AerodynamicsPkg.DragPolar; "
        "FAST-Python still uses the constant-L/D mission model for native "
        "end-to-end sizing."
    ),
    "ERJ175LR_ClimbThenAccel": (
        "MATLAB FAST now flies ERJ175LR with AerodynamicsPkg.DragPolar; "
        "FAST-Python still uses the constant-L/D mission model for native "
        "end-to-end sizing."
    ),
    "ERJ190_FE": (
        "MATLAB FAST reaches Settings.Converged = 0 for this fully electric "
        "sizing case, so tiny regression differences are amplified through "
        "the 50-iteration divergent sizing loop."
    ),
}
MATLAB_REFERENCE_METHOD_ERROR = (
    "MATLAB FAST mainline currently errors during SpecProcessing when legacy "
    "aircraft specs omit Aero.L_D.Method."
)


def selected_case_names():
    """Return requested output-parity cases, defaulting to the full matrix."""

    requested = os.environ.get("FAST_PYTHON_OUTPUT_PARITY_CASES", "").strip()

    if not requested:
        return native_case_names()

    selected = [
        item.strip()
        for item in requested.split(",")
        if item.strip()
    ]
    known = {
        name.lower(): name
        for name in native_case_names()
    }
    unknown = [
        item
        for item in selected
        if item.lower() not in known
    ]

    if unknown:
        supported = ", ".join(native_case_names())
        raise ValueError(
            f"Unknown FAST_PYTHON_OUTPUT_PARITY_CASES item(s): {unknown}. "
            f"Supported cases: {supported}."
        )

    return [
        known[item.lower()]
        for item in selected
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


@pytest.mark.parametrize("case_name", selected_case_names())
def test_output_aircraft_matches_matlab_fast(matlab_wrapper, case_name):
    """Compare one native case OutputAircraft against MATLAB FAST."""

    wrapper, wrapper_build_json_data = matlab_wrapper
    aircraft, mission = native_case(case_name)
    actual = build_json_data(run(deepcopy(aircraft), deepcopy(mission))["aircraft"])
    try:
        expected = matlab_output_aircraft(
            wrapper,
            wrapper_build_json_data,
            case_name,
        )
    except Exception as error:
        if matlab_missing_ld_method_error(error):
            pytest.xfail(MATLAB_REFERENCE_METHOD_ERROR)

        raise
    failures, compared = compare_json_value(actual, expected, "Aircraft")

    assert compared > 0

    if failures:
        if case_name in NONCONVERGED_OUTPUT_CASES:
            pytest.xfail(NONCONVERGED_OUTPUT_CASES[case_name])

        preview = "\n".join(failures[:50])
        pytest.fail(f"{case_name} OutputAircraft parity failures:\n{preview}")


def matlab_missing_ld_method_error(error):
    """Return true for the current MATLAB reference missing-Method failure."""

    text = str(error)
    return (
        'Unrecognized field name "Method"' in text
        and "SpecProcessing" in text
    )


def matlab_output_aircraft(wrapper, wrapper_build_json_data, case_name):
    """Run the canonical MATLAB AircraftSpecsPkg/MissionProfilesPkg pair."""

    aircraft_name, profile_name = OUTPUT_PARITY_MATLAB_CASES[case_name]
    wrapper.engine.evalc(
        f"""
        matlab_aircraft = Main(...
            AircraftSpecsPkg.{aircraft_name}(), ...
            @MissionProfilesPkg.{profile_name});
        if isfield(matlab_aircraft, "HistData")
            matlab_aircraft = rmfield(matlab_aircraft, "HistData");
        end
        if isfield(matlab_aircraft, "RegressionParams")
            matlab_aircraft = rmfield(matlab_aircraft, "RegressionParams");
        end
        if isfield(matlab_aircraft, "Geometry") && ...
           isfield(matlab_aircraft.Geometry, "Preset")
            matlab_aircraft.Geometry = rmfield(matlab_aircraft.Geometry, "Preset");
        end
        if isfield(matlab_aircraft, "Specs") && ...
           isfield(matlab_aircraft.Specs, "Propulsion") && ...
           isfield(matlab_aircraft.Specs.Propulsion, "SizedEngine")
            matlab_aircraft.Specs.Propulsion = ...
                rmfield(matlab_aircraft.Specs.Propulsion, "SizedEngine");
        end
        if isfield(matlab_aircraft, "Specs") && ...
           isfield(matlab_aircraft.Specs, "Aero") && ...
           isfield(matlab_aircraft.Specs.Aero, "L_D") && ...
           isfield(matlab_aircraft.Specs.Aero.L_D, "Method")
            matlab_aircraft.Specs.Aero.L_D = ...
                rmfield(matlab_aircraft.Specs.Aero.L_D, "Method");
        end
        """,
        nargout=1,
    )
    matlab_aircraft = wrapper._to_python_data(wrapper.engine.workspace["matlab_aircraft"])
    return wrapper_build_json_data(matlab_aircraft)
