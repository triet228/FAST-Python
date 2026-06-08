# src/fast_python/core.py

"""Public FAST Python run API."""

from fast_python.aircraft import prepare_aircraft
from fast_python.reference import ReferenceCaseStore, ReferenceDataError


class UnsupportedCaseError(RuntimeError):
    """Report an input case outside the current Python backend coverage."""


class FastPython:
    """Run the Python implementation of FAST's JSON-facing workflow.

    Inputs:
        reference_path: Optional FAST-Python-Wrapper checkout path containing
            examples used by the current reference backend.

    Outputs:
        run() returns a dictionary with status, mtow, aircraft, log, and backend
        metadata matching the wrapper's high-level return contract.

    Assumptions:
        The first backend is a MATLAB-free reference backend for wrapper parity
        cases. It preserves the API that algorithm modules will continue to use
        as the remaining MATLAB packages are ported into native Python.
    """

    def __init__(self, reference_path=None):
        self.reference_store = ReferenceCaseStore(reference_path)

    def run(self, aircraft, mission):
        """Run a supported FAST aircraft and mission case."""

        aircraft = prepare_aircraft(aircraft)

        try:
            case_name, output = self.reference_store.match(aircraft, mission)
        except ReferenceDataError as error:
            raise UnsupportedCaseError(str(error)) from error

        mtow = get_nested(output, ["Specs", "Weight", "MTOW"])

        return {
            "status": "success",
            "mtow": float(mtow),
            "aircraft": output,
            "log": (
                "FAST-Python reference backend returned the saved "
                f"FAST-Python-Wrapper {case_name} OutputAircraft baseline."
            ),
            "backend": "reference",
            "case": case_name,
        }


def run(aircraft, mission, reference_path=None):
    """Run FAST with the default Python backend."""

    return FastPython(reference_path).run(aircraft, mission)


def get_nested(value, keys):
    """Read a nested dictionary field."""

    current = value

    for key in keys:
        current = current[key]

    return current
