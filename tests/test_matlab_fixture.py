# tests/test_matlab_fixture.py

"""Tests for optional MATLAB parity fixture setup."""

from conftest import is_missing_matlab_engine_error


def test_missing_matlab_engine_error_is_detected_from_wrapper_message():
    """Check wrapper startup failures can skip when MATLAB Engine is absent."""

    error = RuntimeError("MATLAB Engine for Python is not installed in this environment.")

    assert is_missing_matlab_engine_error(error)


def test_missing_matlab_engine_error_is_detected_from_exception_cause():
    """Check direct import failures also map to a missing MATLAB Engine skip."""

    cause = ModuleNotFoundError("No module named 'matlab'", name="matlab")

    try:
        raise RuntimeError("wrapper startup failed") from cause
    except RuntimeError as error:
        assert is_missing_matlab_engine_error(error)


def test_non_engine_wrapper_error_is_not_suppressed():
    """Check unrelated MATLAB startup errors still fail parity setup visibly."""

    error = RuntimeError("MATLAB license checkout failed.")

    assert not is_missing_matlab_engine_error(error)
