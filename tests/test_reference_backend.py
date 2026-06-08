# tests/test_reference_backend.py

"""Parity tests for the FAST Python reference backend."""

import pytest

from fast_python import FastPython, UnsupportedCaseError
from fast_python.compare import compare_json_value
from fast_python.reference import (
    CASE_NAMES,
    ReferenceCaseStore,
    load_bundled_case_inputs,
)
from fast_python.main import main


@pytest.fixture
def reference_cases():
    """Return bundled reference cases keyed by case name."""

    return {
        case["name"]: case
        for case in ReferenceCaseStore().cases()
    }


def load_case(reference_cases, case_name):
    """Load one bundled fixture case."""

    aircraft, mission = load_bundled_case_inputs(case_name)
    expected = reference_cases[case_name]["output"]
    return aircraft, mission, expected


@pytest.mark.parametrize("case_name", CASE_NAMES)
def test_reference_backend_matches_wrapper_outputs(reference_cases, case_name):
    """Check Python output against saved wrapper OutputAircraft JSON."""

    aircraft, mission, expected = load_case(reference_cases, case_name)
    result = FastPython().run(aircraft, mission)
    failures, compared = compare_json_value(result["aircraft"], expected)

    assert result["status"] == "success"
    assert result["case"] == case_name
    assert compared > 0
    assert not failures, "\n".join(failures[:50])


def test_reference_backend_rejects_unknown_case(reference_cases):
    """Check that unsupported inputs fail clearly."""

    aircraft, mission, _ = load_case(reference_cases, "A320")
    aircraft["Specs"]["TLAR"]["MaxPax"] = 999

    with pytest.raises(UnsupportedCaseError, match="supports only wrapper fixture"):
        FastPython().run(aircraft, mission)


def test_cli_main_runs_bundled_case(tmp_path):
    """Check that a bundled case can run without wrapper input files."""

    result = main(OUTPUT_DIR=tmp_path, case="A320")

    assert result["status"] == "success"
    assert result["case"] == "A320"
    assert (tmp_path / "OutputAircraft.json").exists()
    assert (tmp_path / "OutputAircraftStructure.json").exists()
