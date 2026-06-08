# src/fast_python/main.py

"""Command-line entry point for FAST Python.

The CLI reads wrapper-style JSON inputs or bundled/native cases, runs the
selected backend, and writes OutputAircraft plus a lightweight structure map.
The function names intentionally preserve FAST's uppercase INPUT_DIR/OUTPUT_DIR
arguments for MATLAB-wrapper familiarity.
"""

import argparse

from fast_python.cases import native_case, native_case_names
from fast_python.core import FastPython
from fast_python.io import (
    build_output_aircraft_structure,
    load_input_json_files,
    save_output_aircraft,
    save_output_aircraft_structure,
)
from fast_python.native import run_native
from fast_python.reference import CASE_NAMES, load_bundled_case_inputs


def main(
    INPUT_DIR="inputs",
    OUTPUT_DIR="outputs",
    reference_path=None,
    case=None,
    backend="reference",
    native_case_name=None,
):
    """Run FAST Python from JSON inputs and save OutputAircraft JSON files.

    Inputs:
        INPUT_DIR: Directory containing InputAircraft.json and Mission.json.
        OUTPUT_DIR: Directory where generated output files are written.
        reference_path: Optional FAST-Python-Wrapper checkout path for parity
            fixture data used by the current backend.
        case: Optional bundled reference case name, such as A320.
        backend: "reference" for fixture parity or "native" for the currently
            ported Python pipeline.
        native_case_name: Optional native aircraft/profile case factory name.

    Outputs:
        The run result dictionary returned by FastPython.run().

    Side effects:
        Writes OutputAircraft.json and OutputAircraftStructure.json.
    """

    if native_case_name:
        aircraft, mission = native_case(native_case_name)
        backend = "native"
    elif case:
        aircraft, mission = load_bundled_case_inputs(case)
    else:
        aircraft, mission = load_input_json_files(INPUT_DIR)

    if backend == "native":
        result = run_native(aircraft, mission)
    else:
        result = FastPython(reference_path).run(aircraft, mission)

    output_aircraft = result["aircraft"]
    output_structure = build_output_aircraft_structure(output_aircraft)

    output_aircraft_path = save_output_aircraft(output_aircraft, OUTPUT_DIR)
    output_structure_path = save_output_aircraft_structure(output_structure, OUTPUT_DIR)

    print(f"Status: {result['status']}")
    print(f"Backend: {result['backend']}")
    print(f"Case: {result.get('case', 'native')}")
    print(f"MTOW: {result['mtow']:.6f} kg")
    print(f"Output saved to {output_aircraft_path}")
    print(f"Full structure saved to {output_structure_path}")

    return result


def cli():
    """Parse command-line arguments and run FAST Python."""

    parser = argparse.ArgumentParser(
        description="Run the Python FAST implementation from JSON inputs."
    )
    parser.add_argument(
        "--input-dir",
        default="inputs",
        help="Directory containing InputAircraft.json and Mission.json.",
    )
    parser.add_argument(
        "--case",
        choices=CASE_NAMES,
        default=None,
        help="Run a bundled reference case instead of reading --input-dir.",
    )
    parser.add_argument(
        "--native-case",
        choices=native_case_names(),
        default=None,
        help="Run a native Python aircraft/profile preset instead of --input-dir.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory where OutputAircraft JSON files are written.",
    )
    parser.add_argument(
        "--wrapper-path",
        default=None,
        help="FAST-Python-Wrapper checkout path for reference fixtures.",
    )
    parser.add_argument(
        "--backend",
        choices=("reference", "native"),
        default="reference",
        help="Execution backend to use.",
    )

    args = parser.parse_args()

    if args.case and args.native_case:
        parser.error("--case and --native-case are mutually exclusive.")

    main(
        args.input_dir,
        args.output_dir,
        args.wrapper_path,
        args.case,
        args.backend,
        args.native_case,
    )


if __name__ == "__main__":
    cli()
