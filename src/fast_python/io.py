# src/fast_python/io.py

"""JSON loading, validation, and output helpers for FAST Python runs.

This layer is responsible for wrapper-compatible serialization details:
reserved marker dictionaries, "NaN" string round-tripping, stable formatting,
and lightweight structural validation. Numerical aircraft feasibility is left
to the native FAST modules rather than duplicated here.
"""

import json
from pathlib import Path

import numpy as np

from fast_python.markers import MatlabExpression, MatlabRow


nan = float("nan")

DEFAULT_INPUT_DIR = Path("inputs")
DEFAULT_OUTPUT_DIR = Path("outputs")
AIRCRAFT_JSON_PATH = Path("InputAircraft.json")
MISSION_JSON_PATH = Path("Mission.json")
OUTPUT_AIRCRAFT_JSON_PATH = Path("OutputAircraft.json")
OUTPUT_AIRCRAFT_STRUCTURE_JSON_PATH = Path("OutputAircraftStructure.json")


class JsonValidationError(ValueError):
    """Report an invalid FAST JSON input or output file."""


def build_input_json_paths(input_dir=None):
    """Return the aircraft and mission input paths for a run.

    Inputs:
        input_dir: Optional directory containing InputAircraft.json and
            Mission.json. Defaults to the local inputs directory.

    Outputs:
        Tuple of aircraft JSON path and mission JSON path.
    """

    if input_dir is None:
        base_path = DEFAULT_INPUT_DIR
    else:
        base_path = Path(input_dir)

    return (
        base_path / AIRCRAFT_JSON_PATH,
        base_path / MISSION_JSON_PATH,
    )


def build_output_json_paths(output_dir=None):
    """Return generated OutputAircraft and structure JSON paths.

    Inputs:
        output_dir: Optional directory for generated files. Defaults to the
            local outputs directory.

    Outputs:
        Tuple of OutputAircraft.json path and OutputAircraftStructure.json path.
    """

    if output_dir is None:
        base_path = DEFAULT_OUTPUT_DIR
    else:
        base_path = Path(output_dir)

    return (
        base_path / OUTPUT_AIRCRAFT_JSON_PATH,
        base_path / OUTPUT_AIRCRAFT_STRUCTURE_JSON_PATH,
    )


def read_raw_json_file(path):
    """Parse a JSON file and include the path in syntax errors.

    Inputs:
        path: JSON file path.

    Outputs:
        Parsed JSON value without marker restoration.

    Assumptions:
        UTF-8 is used for wrapper-compatible JSON files.
    """

    path = Path(path)

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise JsonValidationError(
            f"{path} is not valid JSON: {error.msg} at line "
            f"{error.lineno}, column {error.colno}."
        ) from error


def write_json_file(path, value):
    """Write a JSON file with stable formatting.

    Inputs:
        path: Destination file path.
        value: JSON-serializable data.

    Outputs:
        None.

    Side effects:
        Creates the parent directory and overwrites the JSON file with
        two-space indentation.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def build_json_data(value):
    """Return a JSON-safe FAST data structure.

    Inputs:
        value: Python data containing dictionaries, lists, scalars, NaN values,
            and wrapper-compatible marker objects.

    Outputs:
        JSON-serializable data. NaN values are stored as the string "NaN"
        because standard JSON has no portable NaN literal.

    Assumptions:
        Marker dictionaries reserve keys beginning with "_". Output-only marker
        dictionaries are preserved as ordinary dictionaries so saved wrapper
        OutputAircraft fixtures can be emitted unchanged.
    """

    if isinstance(value, MatlabExpression):
        return {"_matlab_expression": value.value}

    if isinstance(value, MatlabRow):
        return {"_matlab_row": build_json_data(value.value)}

    if isinstance(value, np.ndarray):
        return build_json_data(value.tolist())

    if isinstance(value, np.generic):
        return build_json_data(value.item())

    if isinstance(value, dict):
        return {
            key: build_json_data(item)
            for key, item in value.items()
        }

    if isinstance(value, list) or isinstance(value, tuple):
        return [build_json_data(item) for item in value]

    if isinstance(value, float) and value != value:
        return "NaN"

    if value is None or isinstance(value, str):
        return value

    if isinstance(value, bool) or isinstance(value, int) or isinstance(value, float):
        return value

    return {
        "_python_type": type(value).__name__,
        "_repr": str(value),
    }


def load_json_data(value):
    """Restore wrapper-compatible marker values from parsed JSON.

    Inputs:
        value: Parsed JSON value.

    Outputs:
        Python value with "NaN" converted to float NaN and reserved marker
        dictionaries converted to MatlabExpression or MatlabRow objects.

    Assumptions:
        Marker validation has already run for external JSON inputs.
    """

    if isinstance(value, dict):
        if set(value.keys()) == {"_matlab_expression"}:
            return MatlabExpression(value["_matlab_expression"])

        if set(value.keys()) == {"_matlab_row"}:
            return MatlabRow(load_json_data(value["_matlab_row"]))

        return {
            key: load_json_data(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [load_json_data(item) for item in value]

    if value == "NaN":
        return nan

    return value


def require_json_object(value, path):
    """Return value when it is a JSON object, otherwise fail."""

    if not isinstance(value, dict):
        raise JsonValidationError(f"{path} must be a JSON object.")

    return value


def get_json_path(data, keys, file_name):
    """Read a required nested JSON field."""

    current = data
    walked = []

    for key in keys:
        walked.append(key)

        if not isinstance(current, dict) or key not in current:
            joined = ".".join(walked)
            raise JsonValidationError(f"{file_name} is missing required field {joined}.")

        current = current[key]

    return current


def is_json_number(value):
    """Return True for JSON numeric values that are not booleans."""

    return (isinstance(value, int) or isinstance(value, float)) and not isinstance(
        value,
        bool,
    )


def require_json_number(data, keys, file_name):
    """Validate that a field is numeric or the FAST NaN/input marker."""

    value = get_json_path(data, keys, file_name)

    if is_json_number(value) or value == "NaN":
        return

    if (
        isinstance(value, dict)
        and set(value.keys()) == {"_matlab_expression"}
        and isinstance(value["_matlab_expression"], str)
    ):
        return

    joined = ".".join(keys)
    raise JsonValidationError(
        f"{file_name}.{joined} must be a number, \"NaN\", "
        "or a _matlab_expression marker."
    )


def require_json_string(data, keys, file_name):
    """Validate that a field is a string."""

    value = get_json_path(data, keys, file_name)

    if isinstance(value, str):
        return

    joined = ".".join(keys)
    raise JsonValidationError(f"{file_name}.{joined} must be a string.")


def require_json_list(data, keys, file_name):
    """Validate that a field is a list and return it."""

    value = get_json_path(data, keys, file_name)

    if isinstance(value, list):
        return value

    joined = ".".join(keys)
    raise JsonValidationError(f"{file_name}.{joined} must be a JSON array.")


def read_json_list_or_scalar(data, keys, file_name):
    """Return a JSON field as a list, accepting scalar values."""

    value = get_json_path(data, keys, file_name)

    if isinstance(value, list):
        return value

    return [value]


def validate_json_markers(value, file_name, path="", allow_output_markers=False):
    """Validate reserved marker dictionaries used by wrapper-compatible JSON.

    Inputs:
        value: Parsed JSON subtree.
        file_name: File label used in validation errors.
        path: Dot/bracket path to the current subtree.
        allow_output_markers: True when OutputAircraft _python_type/_repr
            marker dictionaries are allowed.

    Outputs:
        None. Raises JsonValidationError for malformed marker dictionaries.

    Assumptions:
        Keys starting with "_" are reserved so unsupported markers fail early
        instead of being silently treated as ordinary FAST fields.
    """

    if isinstance(value, dict):
        keys = set(value.keys())
        label = path or file_name
        marker_keys = [key for key in keys if key.startswith("_")]

        if marker_keys:
            if keys == {"_matlab_expression"}:
                if not isinstance(value["_matlab_expression"], str):
                    raise JsonValidationError(
                        f"{label}._matlab_expression must be a string."
                    )
                return

            if keys == {"_matlab_row"}:
                if not isinstance(value["_matlab_row"], list):
                    raise JsonValidationError(f"{label}._matlab_row must be an array.")
                validate_json_markers(
                    value["_matlab_row"],
                    file_name,
                    f"{label}._matlab_row",
                    allow_output_markers,
                )
                return

            if allow_output_markers and keys == {"_python_type", "_repr"}:
                if not isinstance(value["_python_type"], str):
                    raise JsonValidationError(f"{label}._python_type must be a string.")
                if not isinstance(value["_repr"], str):
                    raise JsonValidationError(f"{label}._repr must be a string.")
                return

            raise JsonValidationError(f"{label} contains invalid marker keys.")

        for key, item in value.items():
            child_path = key if not path else f"{path}.{key}"
            validate_json_markers(item, file_name, child_path, allow_output_markers)

        return

    if isinstance(value, list):
        for index, item in enumerate(value):
            child_path = f"{path}[{index}]"
            validate_json_markers(item, file_name, child_path, allow_output_markers)


def validate_aircraft_json(data):
    """Validate InputAircraft.json before converting it to Python data.

    Inputs:
        data: Parsed JSON object.

    Outputs:
        None. Raises JsonValidationError when required FAST input fields or
        marker shapes are missing.

    Assumptions:
        This is a structural validation layer, not a complete aircraft sizing
        validation. Numerical feasibility is handled by native FAST modules.
    """

    require_json_object(data, "InputAircraft.json")
    validate_json_markers(data, "InputAircraft.json")

    require_json_string(data, ["Specs", "TLAR", "Class"], "InputAircraft.json")
    require_json_number(data, ["Specs", "TLAR", "MaxPax"], "InputAircraft.json")
    require_json_number(data, ["Specs", "Performance", "Range"], "InputAircraft.json")
    require_json_number(data, ["Specs", "Weight", "MTOW"], "InputAircraft.json")

    get_json_path(data, ["Specs", "Propulsion", "PropArch"], "InputAircraft.json")
    get_json_path(data, ["Settings"], "InputAircraft.json")


def validate_mission_json(data):
    """Validate Mission.json before converting it to Python data.

    Inputs:
        data: Parsed JSON object.

    Outputs:
        None. Raises JsonValidationError for malformed segment arrays, target
        arrays, or marker dictionaries.

    Assumptions:
        Segment arrays must have equal length because process_profile assigns
        one beginning/end point range per segment.
    """

    require_json_object(data, "Mission.json")
    validate_json_markers(data, "Mission.json")

    targets = read_json_list_or_scalar(data, ["Target", "Valu"], "Mission.json")
    target_types = read_json_list_or_scalar(data, ["Target", "Type"], "Mission.json")

    if len(targets) != len(target_types):
        raise JsonValidationError(
            "Mission.json Target.Valu and Target.Type must have the same length."
        )

    for index, target_type in enumerate(target_types):
        if target_type not in ("Dist", "Time"):
            raise JsonValidationError(
                f"Mission.json Target.Type[{index}] must be \"Dist\" or \"Time\"."
            )

    segment_fields = [
        "Segs",
        "ID",
        "AltBeg",
        "AltEnd",
        "VelBeg",
        "VelEnd",
        "TypeBeg",
        "TypeEnd",
        "ClbRate",
    ]
    segment_lengths = {}

    for field_name in segment_fields:
        segment_lengths[field_name] = len(
            require_json_list(data, [field_name], "Mission.json")
        )

    expected_length = segment_lengths["Segs"]

    for field_name, length in segment_lengths.items():
        if length != expected_length:
            raise JsonValidationError(
                "Mission.json segment arrays must have the same length: "
                f"Segs has {expected_length}, {field_name} has {length}."
            )

    for index, segment_name in enumerate(data["Segs"]):
        if not isinstance(segment_name, str):
            raise JsonValidationError(f"Mission.json Segs[{index}] must be a string.")


def validate_output_aircraft_json(data):
    """Validate generated OutputAircraft.json."""

    require_json_object(data, "OutputAircraft.json")
    validate_json_markers(data, "OutputAircraft.json", allow_output_markers=True)

    require_json_number(data, ["Specs", "Weight", "MTOW"], "OutputAircraft.json")
    require_json_number(data, ["Specs", "Weight", "Fuel"], "OutputAircraft.json")
    require_json_number(data, ["Specs", "Aero", "S"], "OutputAircraft.json")
    require_json_string(data, ["Specs", "TLAR", "Class"], "OutputAircraft.json")
    get_json_path(data, ["Mission", "Profile"], "OutputAircraft.json")


def validate_output_structure_json(data):
    """Validate generated OutputAircraftStructure.json."""

    require_json_object(data, "OutputAircraftStructure.json")

    for field_name in ("Specs", "Mission"):
        if field_name not in data:
            raise JsonValidationError(
                f"OutputAircraftStructure.json is missing required field {field_name}."
            )


def read_json_file(path, validator=None):
    """Read and validate a JSON file, then restore marker values.

    Inputs:
        path: JSON file path.
        validator: Optional validation callable that accepts parsed JSON data.

    Outputs:
        Runtime Python data with wrapper markers restored.
    """

    data = read_raw_json_file(path)

    if validator:
        validator(data)

    return load_json_data(data)


def require_input_json_file(path):
    """Fail when a required FAST JSON input file is missing."""

    path = Path(path)

    if not path.exists():
        raise JsonValidationError(
            f"{path} is required. Provide this input file, then rerun fast-python."
        )


def load_input_json_files(input_dir=None):
    """Load InputAircraft.json and Mission.json from an input directory.

    Inputs:
        input_dir: Optional input directory path.

    Outputs:
        Tuple of aircraft dictionary and mission dictionary.

    Side effects:
        Reads both required JSON files and raises JsonValidationError before a
        run starts if either file is missing or invalid.
    """

    aircraft_json_path, mission_json_path = build_input_json_paths(input_dir)

    require_input_json_file(aircraft_json_path)
    require_input_json_file(mission_json_path)

    return (
        read_json_file(aircraft_json_path, validate_aircraft_json),
        read_json_file(mission_json_path, validate_mission_json),
    )


def build_output_aircraft_structure(value):
    """Return a recursive structure map for OutputAircraft data.

    Inputs:
        value: OutputAircraft subtree.

    Outputs:
        Dictionary/list/type-name structure description suitable for
        OutputAircraftStructure.json.

    Assumptions:
        For lists, the first item represents the element structure. This mirrors
        the wrapper's lightweight schema output rather than a full JSON Schema.
    """

    if isinstance(value, dict):
        return {
            key: build_output_aircraft_structure(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        item_structure = None

        if value:
            item_structure = build_output_aircraft_structure(value[0])

        return {
            "_type": "list",
            "_length": len(value),
            "_items": item_structure,
        }

    return type(value).__name__


def save_output_aircraft(value, output_dir=None):
    """Write OutputAircraft.json and return its path."""

    output_aircraft_path, _ = build_output_json_paths(output_dir)
    write_json_file(output_aircraft_path, build_json_data(value))
    validate_output_aircraft_json(read_raw_json_file(output_aircraft_path))
    return output_aircraft_path


def save_output_aircraft_structure(value, output_dir=None):
    """Write OutputAircraftStructure.json and return its path."""

    _, output_structure_path = build_output_json_paths(output_dir)
    write_json_file(output_structure_path, value)
    validate_output_structure_json(read_raw_json_file(output_structure_path))
    return output_structure_path
