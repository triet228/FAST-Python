# src/fast_python/reference.py

"""Reference-case backend used while MATLAB algorithms are ported to Python."""

import hashlib
import json
import os
import zipfile
from copy import deepcopy
from pathlib import Path

from fast_python.aircraft import prepare_aircraft
from fast_python.io import (
    build_json_data,
    load_json_data,
    read_json_file,
    read_raw_json_file,
    validate_aircraft_json,
    validate_mission_json,
)


CASE_NAMES = ("A320", "AEA", "ATR42", "CeRAS")


class ReferenceDataError(RuntimeError):
    """Report missing or unusable FAST-Python-Wrapper reference fixtures."""


class ReferenceCaseStore:
    """Load and match wrapper fixture cases.

    Inputs:
        wrapper_path: Optional path to FAST-Python-Wrapper. If missing, common
            local paths and FAST_PYTHON_WRAPPER_PATH are tried.

    Outputs:
        A store whose match() method returns saved OutputAircraft data for a
        known aircraft and mission input pair.

    Assumptions:
        This backend is an explicit bridge while MATLAB algorithms are ported.
        It is deterministic and MATLAB-free, but it only supports fixture cases
        that have a saved wrapper OutputAircraft.json.
    """

    def __init__(self, wrapper_path=None):
        self.wrapper_path = None
        self.examples_path = None
        self.archive_path = None

        if wrapper_path:
            self.wrapper_path = resolve_wrapper_path(wrapper_path)
            self.examples_path = self.wrapper_path / "examples"
        else:
            archive_path = bundled_reference_archive()

            if archive_path.exists():
                self.archive_path = archive_path
            else:
                self.wrapper_path = resolve_wrapper_path()
                self.examples_path = self.wrapper_path / "examples"

        self._cases = None

    def match(self, aircraft, mission):
        """Return the case name and saved output for a supported input pair."""

        key = make_case_key(aircraft, mission)

        for case in self.cases():
            if key in case["keys"]:
                return case["name"], deepcopy(case["output"])

        supported = ", ".join(case["name"] for case in self.cases())
        raise ReferenceDataError(
            "This Python backend currently supports only wrapper fixture "
            f"inputs: {supported}. Add or port a backend for this aircraft and "
            "mission before running arbitrary FAST cases."
        )

    def cases(self):
        """Return cached wrapper reference cases."""

        if self._cases is None:
            if self.archive_path:
                self._cases = load_reference_cases_from_archive(self.archive_path)
            else:
                self._cases = load_reference_cases_from_directory(self.examples_path)

        return self._cases


def bundled_reference_archive():
    """Return the bundled reference archive path."""

    return Path(__file__).resolve().parent / "data" / "reference_cases.zip"


def resolve_wrapper_path(wrapper_path=None):
    """Return the FAST-Python-Wrapper path used for reference fixtures."""

    candidates = []

    if wrapper_path:
        candidates.append(Path(wrapper_path))

    env_path = os.environ.get("FAST_PYTHON_WRAPPER_PATH")
    if env_path:
        candidates.append(Path(env_path))

    repo_root = Path(__file__).resolve().parents[2]
    candidates.extend(
        [
            repo_root.parent / "FAST-Python-Wrapper",
            Path.home() / "Projects" / "FAST-Python-Wrapper",
            Path("C:/Users/homin/Projects/FAST-Python-Wrapper"),
        ]
    )

    for candidate in candidates:
        path = candidate.expanduser().resolve()

        if (path / "examples").is_dir():
            return path

    checked = ", ".join(str(item) for item in candidates)
    raise ReferenceDataError(
        "FAST-Python-Wrapper examples were not found. Set "
        f"FAST_PYTHON_WRAPPER_PATH. Checked: {checked}"
    )


def load_reference_cases_from_directory(examples_path):
    """Load known wrapper fixture inputs and saved OutputAircraft outputs."""

    cases = []

    for name in CASE_NAMES:
        case_path = examples_path / name
        aircraft_path = case_path / "inputs" / "InputAircraft.json"
        mission_path = case_path / "inputs" / "Mission.json"
        output_path = case_path / "outputs" / "OutputAircraft.json"

        if not aircraft_path.exists() or not mission_path.exists() or not output_path.exists():
            raise ReferenceDataError(f"Reference case {name} is incomplete: {case_path}")

        aircraft = read_json_file(aircraft_path, validate_aircraft_json)
        mission = read_json_file(mission_path, validate_mission_json)
        output = read_raw_json_file(output_path)

        cases.append(
            {
                "name": name,
                "keys": {
                    make_case_key(aircraft, mission),
                    make_case_key(prepare_aircraft(aircraft), mission),
                },
                "output": output,
            }
        )

    return cases


def load_reference_cases_from_archive(archive_path):
    """Load known fixture inputs and outputs from the bundled zip archive."""

    cases = []

    with zipfile.ZipFile(archive_path) as archive:
        for name in CASE_NAMES:
            aircraft = read_json_from_archive(
                archive,
                f"{name}/inputs/InputAircraft.json",
                validate_aircraft_json,
                restore_markers=True,
            )
            mission = read_json_from_archive(
                archive,
                f"{name}/inputs/Mission.json",
                validate_mission_json,
                restore_markers=True,
            )
            output = read_json_from_archive(
                archive,
                f"{name}/outputs/OutputAircraft.json",
            )

            cases.append(
                {
                    "name": name,
                    "keys": {
                        make_case_key(aircraft, mission),
                        make_case_key(prepare_aircraft(aircraft), mission),
                    },
                    "output": output,
                }
            )

    return cases


def load_bundled_case_inputs(case_name):
    """Return aircraft and mission inputs for a bundled reference case."""

    if case_name not in CASE_NAMES:
        supported = ", ".join(CASE_NAMES)
        raise ReferenceDataError(
            f"Unknown bundled case {case_name!r}. Supported cases: {supported}."
        )

    archive_path = bundled_reference_archive()

    if not archive_path.exists():
        raise ReferenceDataError(f"Bundled reference archive not found: {archive_path}")

    with zipfile.ZipFile(archive_path) as archive:
        aircraft = read_json_from_archive(
            archive,
            f"{case_name}/inputs/InputAircraft.json",
            validate_aircraft_json,
            restore_markers=True,
        )
        mission = read_json_from_archive(
            archive,
            f"{case_name}/inputs/Mission.json",
            validate_mission_json,
            restore_markers=True,
        )

    return aircraft, mission


def read_json_from_archive(archive, member_name, validator=None, restore_markers=False):
    """Read one JSON member from the reference archive."""

    try:
        raw = archive.read(member_name)
    except KeyError as error:
        raise ReferenceDataError(f"Reference archive is missing {member_name}.") from error

    data = json.loads(raw.decode("utf-8"))

    if validator:
        validator(data)

    if restore_markers:
        return load_json_data(data)

    return data


def make_case_key(aircraft, mission):
    """Return a stable fingerprint for an aircraft and mission input pair."""

    payload = {
        "aircraft": build_json_data(aircraft),
        "mission": build_json_data(mission),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
