# tests/conftest.py

"""Shared pytest fixtures for optional MATLAB-backed parity tests."""

import os
import sys
from pathlib import Path

import pytest


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

    wrapper = CurrentWrapperBridge(wrapper_path, fast_path)
    try:
        wrapper.start()
    except RuntimeError as error:
        if is_missing_matlab_engine_error(error):
            pytest.skip(str(error))
        raise

    try:
        yield wrapper, wrapper.build_json_data
    finally:
        wrapper.stop()


class CurrentWrapperBridge:
    """Expose the small wrapper surface used by FAST-Python parity tests."""

    def __init__(self, wrapper_path, fast_path):
        self.wrapper_path = wrapper_path
        self.fast_path = fast_path
        self.engine = None
        self.build_json_data = None
        self._matlab_to_python = None
        self._python_to_matlab = None
        self._resolve_fast_path = None
        self._start_matlab = None

    def start(self):
        """Start MATLAB through the current FAST-Python-Wrapper core API."""

        self._load_wrapper_api()
        self.engine = self._start_matlab(self._resolve_fast_path(self.fast_path))
        self.engine.addpath(str(self.wrapper_path / "core"), nargout=0)

    def stop(self):
        """Quit MATLAB if this fixture started it."""

        if self.engine is not None:
            self.engine.quit()
            self.engine = None

    def _to_matlab_literal(self, value):
        """Return a MATLAB literal for wrapper-compatible Python data."""

        return self._python_to_matlab(value)

    def _to_python_data(self, value):
        """Return Python data converted from a MATLAB Engine value."""

        return self._matlab_to_python(value)

    def _load_wrapper_api(self):
        """Import FAST-Python-Wrapper modules from the configured checkout."""

        wrapper_path_text = str(self.wrapper_path)

        if wrapper_path_text not in sys.path:
            sys.path.insert(0, wrapper_path_text)

        try:
            from core.json_io import build_json_data
            from core.matlab_bridge import (
                matlab_to_python,
                python_to_matlab,
                resolve_fast_path,
                start_matlab,
            )
        except ModuleNotFoundError as error:
            pytest.skip(f"FAST-Python-Wrapper import failed: {error}")

        self.build_json_data = build_json_data
        self._matlab_to_python = matlab_to_python
        self._python_to_matlab = python_to_matlab
        self._resolve_fast_path = resolve_fast_path
        self._start_matlab = start_matlab


def is_missing_matlab_engine_error(error):
    """Return true when wrapper startup failed because MATLAB Engine is absent."""

    if "MATLAB Engine for Python is not installed" in str(error):
        return True

    cause = getattr(error, "__cause__", None)
    while cause is not None:
        if isinstance(cause, ModuleNotFoundError) and cause.name == "matlab":
            return True
        cause = getattr(cause, "__cause__", None)

    return False
