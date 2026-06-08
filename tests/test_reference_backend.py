# tests/test_reference_backend.py

"""Parity tests for the FAST Python reference backend."""

import pytest

from fast_python import FastPython, UnsupportedCaseError
from fast_python.compare import compare_json_value
from fast_python.io import (
    read_json_file,
    read_raw_json_file,
    validate_aircraft_json,
    validate_mission_json,
)
from fast_python.reference import CASE_NAMES, resolve_wrapper_path
from fast_python.main import main


@pytest.fixture
def examples_path():
    """Return the wrapper examples path used as the parity oracle."""

    return resolve_wrapper_path() / "examples"


def load_case(examples_path, case_name):
    """Load one wrapper fixture case."""

    case_path = examples_path / case_name
    aircraft = read_json_file(
        case_path / "inputs" / "InputAircraft.json",
        validate_aircraft_json,
    )
    mission = read_json_file(
        case_path / "inputs" / "Mission.json",
        validate_mission_json,
    )
    expected = read_raw_json_file(case_path / "outputs" / "OutputAircraft.json")
    return aircraft, mission, expected


@pytest.mark.parametrize("case_name", CASE_NAMES)
def test_reference_backend_matches_wrapper_outputs(examples_path, case_name):
    """Check Python output against saved wrapper OutputAircraft JSON."""

    aircraft, mission, expected = load_case(examples_path, case_name)
    result = FastPython().run(aircraft, mission)
    failures, compared = compare_json_value(result["aircraft"], expected)

    assert result["status"] == "success"
    assert result["case"] == case_name
    assert compared > 0
    assert not failures, "\n".join(failures[:50])


def test_reference_backend_rejects_unknown_case(examples_path):
    """Check that unsupported inputs fail clearly."""

    aircraft, mission, _ = load_case(examples_path, "A320")
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
