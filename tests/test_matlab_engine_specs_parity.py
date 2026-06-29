# tests/test_matlab_engine_specs_parity.py

"""Optional MATLAB parity tests for every EngineModelPkg.EngineSpecsPkg preset."""

import pytest

from fast_python import specs
from fast_python.compare import compare_json_value
from fast_python.io import build_json_data


ENGINE_SPEC_CASES = [
    ("AE2100_D3", specs.ae2100_d3),
    ("AE3007A", specs.ae3007a),
    ("AE501D_22G", specs.ae501d_22g),
    ("Allison_250_C30G", specs.allison_250_c30g),
    ("CeRAS", specs.ceras_engine),
    ("CF6_80C2_B7F", specs.cf6_80c2_b7f),
    ("CF34_8E5", specs.cf34_8e5),
    ("ExampleTF", specs.example_turbofan),
    ("ExampleTP", specs.example_turboprop),
    ("LEAP_1A26", specs.leap_1a26),
    ("PT6A_114A", specs.pt6a_114a),
    ("PW_123", specs.pw_123),
    ("PW_127M", specs.pw_127m),
    ("PW_1919G", specs.pw_1919g),
    ("PW_2037", specs.pw_2037),
    ("RB211_22B_02", specs.rb211_22b_02),
    ("TPE331_14GR_805H", specs.tpe331_14gr_805h),
    ("Trent_970B_84", specs.trent_970b_84),
]

@pytest.mark.parametrize("matlab_name, python_factory", ENGINE_SPEC_CASES)
def test_engine_specs_pkg_matches_matlab(matlab_wrapper, matlab_name, python_factory):
    """Compare one EngineSpecsPkg preset against MATLAB FAST output."""

    wrapper, wrapper_build_json_data = matlab_wrapper
    expected = matlab_engine_spec(wrapper, wrapper_build_json_data, matlab_name)
    actual = build_json_data(python_factory())
    failures, compared = compare_json_value(actual, expected, "Engine")

    assert compared > 0

    if failures:
        preview = "\n".join(failures[:25])
        pytest.fail(f"{matlab_name} parity failures:\n{preview}")


def matlab_engine_spec(wrapper, wrapper_build_json_data, matlab_name):
    """Return one MATLAB EngineSpecsPkg output as JSON-comparable data."""

    wrapper.engine.evalc(
        f"""
        matlab_engine = EngineModelPkg.EngineSpecsPkg.{matlab_name}();
        """,
        nargout=1,
    )
    matlab_engine = wrapper._to_python_data(wrapper.engine.workspace["matlab_engine"])
    return wrapper_build_json_data(matlab_engine)
