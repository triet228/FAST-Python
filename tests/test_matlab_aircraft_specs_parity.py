# tests/test_matlab_aircraft_specs_parity.py

"""Optional MATLAB parity tests for every AircraftSpecsPkg aircraft preset."""

import inspect

import numpy as np
import pytest

from fast_python import specs
from fast_python.compare import compare_json_value
from fast_python.io import build_json_data


SPLIT_SAMPLE = 0.2
AIRCRAFT_SPEC_CASES = [
    ("A320Neo", specs.a320neo),
    ("AEA", specs.aea),
    ("ATR42", specs.atr42),
    ("CeRAS", specs.aircraft_ceras),
    ("ERJ175LR", specs.erj175lr),
    ("ERJ175LR_Elec", specs.erj175lr_elec),
    ("ERJ190_E2", specs.erj190_e2),
    ("ERJ190_FE", specs.erj190_fe),
    ("Example", specs.example_aircraft),
    ("LM100J_Conventional", specs.lm100j_conventional),
    ("LM100J_Hybrid", specs.lm100j_hybrid),
]

@pytest.mark.parametrize("matlab_name, python_factory", AIRCRAFT_SPEC_CASES)
def test_aircraft_specs_pkg_matches_matlab(matlab_wrapper, matlab_name, python_factory):
    """Compare one full AircraftSpecsPkg preset against MATLAB FAST output."""

    wrapper, wrapper_build_json_data = matlab_wrapper
    expected = matlab_aircraft_spec(wrapper, wrapper_build_json_data, matlab_name)
    actual = build_json_data(normalize_python_aircraft(python_factory()))
    failures, compared = compare_json_value(actual, expected, "Aircraft")

    assert compared > 0

    if failures:
        preview = "\n".join(failures[:25])
        pytest.fail(f"{matlab_name} parity failures:\n{preview}")


def matlab_aircraft_spec(wrapper, wrapper_build_json_data, matlab_name):
    """Return one MATLAB AircraftSpecsPkg output as JSON-comparable data."""

    wrapper.engine.evalc(
        f"""
        matlab_aircraft = AircraftSpecsPkg.{matlab_name}();
        if isfield(matlab_aircraft, "Specs") && ...
           isfield(matlab_aircraft.Specs, "Propulsion") && ...
           isfield(matlab_aircraft.Specs.Propulsion, "PropArch")

            if isfield(matlab_aircraft.Specs.Propulsion.PropArch, "OperUps") && ...
               isa(matlab_aircraft.Specs.Propulsion.PropArch.OperUps, "function_handle")

                if nargin(matlab_aircraft.Specs.Propulsion.PropArch.OperUps) == 0
                    matlab_aircraft.Specs.Propulsion.PropArch.OperUps = ...
                        matlab_aircraft.Specs.Propulsion.PropArch.OperUps();
                else
                    matlab_aircraft.Specs.Propulsion.PropArch.OperUps = ...
                        matlab_aircraft.Specs.Propulsion.PropArch.OperUps({SPLIT_SAMPLE});
                end
            end

            if isfield(matlab_aircraft.Specs.Propulsion.PropArch, "OperDwn") && ...
               isa(matlab_aircraft.Specs.Propulsion.PropArch.OperDwn, "function_handle")

                if nargin(matlab_aircraft.Specs.Propulsion.PropArch.OperDwn) == 0
                    matlab_aircraft.Specs.Propulsion.PropArch.OperDwn = ...
                        matlab_aircraft.Specs.Propulsion.PropArch.OperDwn();
                else
                    matlab_aircraft.Specs.Propulsion.PropArch.OperDwn = ...
                        matlab_aircraft.Specs.Propulsion.PropArch.OperDwn({SPLIT_SAMPLE});
                end
            end
        end
        """,
        nargout=1,
    )
    matlab_aircraft = wrapper._to_python_data(wrapper.engine.workspace["matlab_aircraft"])
    return wrapper_build_json_data(matlab_aircraft)


def normalize_python_aircraft(value):
    """Materialize Python-only callables and NumPy objects before comparison."""

    if callable(value):
        try:
            count = len(inspect.signature(value).parameters)
        except (TypeError, ValueError):
            count = 0

        if count == 0:
            return normalize_python_aircraft(value())

        return normalize_python_aircraft(value(SPLIT_SAMPLE))

    if isinstance(value, np.ndarray):
        return normalize_python_aircraft(value.tolist())

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, dict):
        return {
            key: normalize_python_aircraft(item)
            for key, item in value.items()
        }

    if isinstance(value, list) or isinstance(value, tuple):
        return [normalize_python_aircraft(item) for item in value]

    return value
