# src/fast_python/compare.py

"""Output comparison helpers for FAST parity checks.

The reference tests compare Python dictionaries against wrapper-generated JSON
where MATLAB row/column vectors, NaN values, and saved output markers can all
represent equivalent data in different shapes. This module keeps those rules in
one place so numerical tolerances and ignored metadata paths are explicit.
"""

import ast
import math

import numpy as np

from fast_python.markers import MatlabExpression, MatlabRow


IGNORED_OUTPUT_PATHS = {
    "Aircraft.Geometry.Preset",
    "Aircraft.HistData",
    "Aircraft.Mission.ProfileFxn",
    "Aircraft.RegressionParams",
    "Aircraft.Settings.Dir",
    "Aircraft.Settings.OEWIterations",
    "Aircraft.Specs.Aero.L_D.CrsMAC",
    "Aircraft.Specs.Power.P_W.AC",
    "Aircraft.Specs.Propulsion.PropArch.OperDwn",
    "Aircraft.Specs.Propulsion.PropArch.OperUps",
    "Aircraft.Settings.Dir.Size",
    "Aircraft.Settings.Plotting",
}
IGNORED_MISSING_OUTPUT_PATHS = {
    "Aircraft.Specs.Power.Eta.Propeller",
    "Aircraft.Specs.Propulsion.Eta.Therm",
}
UNPARSED_REPR = object()


def compare_json_value(actual, expected, path="Aircraft"):
    """Return recursive JSON comparison failures.

    Inputs:
        actual: JSON-safe Python FAST output.
        expected: Saved wrapper OutputAircraft JSON data.
        path: Dotted path used in failure messages.

    Outputs:
        A tuple of failure messages and scalar/container values compared.

    Assumptions:
        Numeric FAST output can vary slightly between MATLAB/Python and across
        MATLAB releases, so numeric leaves are compared with the wrapper test
        tolerance. Machine-specific paths and plotting handles are ignored.
    """

    actual = normalize_comparison_value(actual)
    expected = normalize_comparison_value(expected)

    if is_ignored_path(path):
        return [], 0

    if isinstance(actual, dict) and isinstance(expected, dict):
        if set(actual) == {"_python_type", "_repr"} and set(expected) == {
            "_python_type",
            "_repr",
        }:
            return compare_output_marker(actual, expected, path)

        failures = []
        compared = 1
        actual_keys = set(actual)
        expected_keys = set(expected)

        for key in sorted(expected_keys - actual_keys):
            child_path = f"{path}.{key}"

            if (
                not is_ignored_path(child_path)
                and not is_ignored_missing_path(child_path)
                and not is_ignorable_missing_value(expected[key])
            ):
                failures.append(f"{child_path} missing from actual output")

        for key in sorted(actual_keys - expected_keys):
            child_path = f"{path}.{key}"

            if (
                not is_ignored_path(child_path)
                and not is_ignored_missing_path(child_path)
                and not is_ignorable_missing_value(actual[key])
            ):
                failures.append(f"{child_path} missing from expected output")

        for key in sorted(actual_keys & expected_keys):
            child_failures, child_compared = compare_json_value(
                actual[key],
                expected[key],
                f"{path}.{key}",
            )
            failures.extend(child_failures)
            compared += child_compared

        return failures, compared

    if isinstance(actual, list) and isinstance(expected, list):
        actual_row = collapse_singleton_row_vector(actual)
        expected_row = collapse_singleton_row_vector(expected)

        if actual_row is not actual or expected_row is not expected:
            return compare_json_value(actual_row, expected_row, path)

        actual_collapsed = collapse_singleton_column_vector(actual)
        expected_collapsed = collapse_singleton_column_vector(expected)

        if actual_collapsed is not actual or expected_collapsed is not expected:
            return compare_json_value(actual_collapsed, expected_collapsed, path)

        failures = []
        compared = 1

        if len(actual) != len(expected):
            return [f"{path} list length mismatch: {len(actual)} != {len(expected)}"], compared

        for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            child_failures, child_compared = compare_json_value(
                actual_item,
                expected_item,
                f"{path}[{index}]",
            )
            failures.extend(child_failures)
            compared += child_compared

        return failures, compared

    if isinstance(actual, list) and len(actual) == 1 and not isinstance(expected, list):
        return compare_json_value(actual[0], expected, path)

    if isinstance(expected, list) and len(expected) == 1 and not isinstance(actual, list):
        return compare_json_value(actual, expected[0], path)

    if is_output_marker(actual):
        return compare_marker_to_value(actual, expected, path)

    if is_output_marker(expected):
        return compare_marker_to_value(expected, actual, path)

    if is_json_number(actual) and is_json_number(expected):
        return compare_numbers(actual, expected, path)

    if actual == expected:
        return [], 1

    return [f"{path} value mismatch: {actual!r} != {expected!r}"], 1


def compare_numbers(actual, expected, path):
    """Compare numeric leaves with FAST parity tolerance."""

    if math.isnan(actual) and math.isnan(expected):
        return [], 1

    if math.isinf(actual) or math.isinf(expected):
        if actual == expected:
            return [], 1

        return [f"{path} numeric mismatch: {actual!r} != {expected!r}"], 1

    tolerance = 1e-6 + 1e-8 * abs(expected)

    if abs(actual - expected) <= tolerance:
        return [], 1

    return [f"{path} numeric mismatch: {actual!r} != {expected!r}"], 1


def normalize_comparison_value(value):
    """Return NumPy values as JSON-comparable Python values."""

    if isinstance(value, MatlabRow):
        return [value.value]

    if isinstance(value, MatlabExpression):
        return value.value

    if isinstance(value, dict) and set(value) == {"_matlab_row"}:
        return [value["_matlab_row"]]

    if isinstance(value, dict) and set(value) == {"_matlab_expression"}:
        return value["_matlab_expression"]

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    if value == "NaN":
        return math.nan

    return value


def collapse_singleton_row_vector(value):
    """Return a flat vector for MATLAB-style 1xN JSON lists."""

    if len(value) != 1 or not isinstance(value[0], list):
        return value

    return value[0]


def collapse_singleton_column_vector(value):
    """Return a flat vector for MATLAB-style Nx1 JSON lists."""

    if not isinstance(value, list) or not value:
        return value

    if not all(isinstance(item, list) and len(item) == 1 for item in value):
        return value

    return [item[0] for item in value]


def is_ignored_path(path):
    """Return True when a comparison path is intentionally ignored."""

    for ignored in IGNORED_OUTPUT_PATHS:
        if path == ignored or path.startswith(f"{ignored}."):
            return True

    return False


def is_ignored_missing_path(path):
    """Return True when only missing/extra comparisons should be ignored."""

    for ignored in IGNORED_MISSING_OUTPUT_PATHS:
        if path == ignored or path.startswith(f"{ignored}."):
            return True

    return False


def is_json_number(value):
    """Return True for JSON numeric values that are not booleans."""

    return (isinstance(value, int) or isinstance(value, float)) and not isinstance(
        value,
        bool,
    )


def is_ignorable_missing_value(value):
    """Return True when a missing/extra field is only default metadata."""

    value = normalize_comparison_value(value)

    if value is None:
        return True

    if is_json_number(value):
        return math.isnan(value) or value == 0

    if is_output_marker(value) and value["_python_type"] == "object":
        return True

    if isinstance(value, dict):
        return all(is_ignorable_missing_value(item) for item in value.values())

    if isinstance(value, list):
        return all(is_ignorable_missing_value(item) for item in value)

    return False


def compare_output_marker(actual, expected, path):
    """Compare saved output marker dictionaries."""

    if actual["_python_type"] != expected["_python_type"]:
        return [
            f"{path} Python type mismatch: "
            f"{actual['_python_type']!r} != {expected['_python_type']!r}"
        ], 1

    if actual["_python_type"] == "object":
        return [], 1

    actual_repr = parse_comparable_repr(actual["_repr"])
    expected_repr = parse_comparable_repr(expected["_repr"])

    if actual_repr is not UNPARSED_REPR and expected_repr is not UNPARSED_REPR:
        return compare_json_value(actual_repr, expected_repr, f"{path}._repr")

    if actual["_repr"] == expected["_repr"]:
        return [], 1

    return [
        f"{path} repr mismatch: "
        f"{actual['_repr'][:120]!r} != {expected['_repr'][:120]!r}"
    ], 1


def is_output_marker(value):
    """Return True for saved Python output marker dictionaries."""

    return isinstance(value, dict) and set(value) == {"_python_type", "_repr"}


def compare_marker_to_value(marker, value, path):
    """Compare a marker dictionary with an already materialized value."""

    if marker["_python_type"] == "object":
        return [], 1

    parsed = parse_comparable_repr(marker["_repr"])

    if parsed is UNPARSED_REPR:
        if marker == value:
            return [], 1

        return [
            f"{path} marker mismatch: {marker['_repr'][:120]!r} != {value!r}"
        ], 1

    return compare_json_value(parsed, value, f"{path}._repr")


def parse_comparable_repr(value):
    """Return a Python value when an output repr is safe to compare."""

    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        parsed = parse_repr_expression(value)

    if not is_comparable_repr_value(parsed):
        return UNPARSED_REPR

    return parsed


def parse_repr_expression(value):
    """Parse repr lists containing MATLAB-style nan or inf tokens."""

    try:
        tree = ast.parse(value, mode="eval")
    except SyntaxError:
        return UNPARSED_REPR

    try:
        return parse_repr_node(tree.body)
    except ValueError:
        return UNPARSED_REPR


def parse_repr_node(node):
    """Return a value from a narrow literal-only repr syntax tree."""

    if isinstance(node, ast.List):
        return [parse_repr_node(item) for item in node.elts]

    if isinstance(node, ast.Tuple):
        return tuple(parse_repr_node(item) for item in node.elts)

    if isinstance(node, ast.Constant):
        return node.value

    if isinstance(node, ast.Name):
        if node.id == "nan":
            return math.nan

        if node.id == "inf":
            return math.inf

    if isinstance(node, ast.UnaryOp):
        value = parse_repr_node(node.operand)

        if isinstance(node.op, ast.USub) and is_json_number(value):
            return -value

        if isinstance(node.op, ast.UAdd) and is_json_number(value):
            return value

    raise ValueError("Unsupported repr node.")


def is_comparable_repr_value(value):
    """Return True when a parsed repr contains only comparable values."""

    if value is None or isinstance(value, str) or isinstance(value, bool):
        return True

    if is_json_number(value):
        return True

    if isinstance(value, list) or isinstance(value, tuple):
        return all(is_comparable_repr_value(item) for item in value)

    return False
