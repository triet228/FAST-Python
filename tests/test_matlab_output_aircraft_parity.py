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
    matlab_result = wrapper.run(deepcopy(aircraft), deepcopy(mission))
    expected = wrapper_build_json_data(matlab_result["aircraft"])
    failures, compared = compare_json_value(actual, expected, "Aircraft")

    assert compared > 0

    if failures:
        preview = "\n".join(failures[:50])
        pytest.fail(f"{case_name} OutputAircraft parity failures:\n{preview}")
