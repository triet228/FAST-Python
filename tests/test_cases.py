# tests/test_cases.py

"""Tests for native FAST baseline case factories."""

import pytest

from fast_python.cases import (
    native_case,
    native_case_aircraft,
    native_case_mission,
    native_case_names,
)


def test_native_case_names_returns_wrapper_baselines():
    """Check the supported native case list."""

    assert native_case_names() == ["A320", "AEA", "ATR42", "CeRAS"]


def test_native_case_pairs_aircraft_specs_and_mission_profiles():
    """Check each native baseline returns the expected aircraft and mission."""

    a320_aircraft, a320_mission = native_case("A320")
    aea_aircraft, aea_mission = native_case("aea")
    atr_aircraft, atr_mission = native_case("ATR42")
    ceras_aircraft, ceras_mission = native_case("CeRAS")

    assert a320_aircraft["Specs"]["Propulsion"]["Engine"]["OPR"] == 50
    assert a320_mission["Segs"][0] == "Takeoff"
    assert aea_aircraft["Specs"]["Propulsion"]["PropArch"]["Type"] == "O"
    assert aea_mission["Target"]["Valu"][0] == 926000
    assert atr_aircraft["Specs"]["TLAR"]["Class"] == "Turboprop"
    assert atr_mission["Target"]["Valu"][2] == 45
    assert ceras_aircraft["Specs"]["Propulsion"]["Engine"]["OPR"] == 24.5
    assert ceras_mission["Segs"] == ["Climb", "Cruise", "Climb", "Cruise", "Descent"]


def test_native_case_singletons_return_defensive_copies():
    """Check aircraft-only and mission-only helpers return independent data."""

    aircraft = native_case_aircraft("A320")
    mission = native_case_mission("A320")
    aircraft["Specs"]["TLAR"]["Class"] = "Changed"
    mission["Segs"][0] = "Changed"

    fresh_aircraft = native_case_aircraft("A320")
    fresh_mission = native_case_mission("A320")

    assert fresh_aircraft["Specs"]["TLAR"]["Class"] == "Turbofan"
    assert fresh_mission["Segs"][0] == "Takeoff"


def test_native_case_rejects_unknown_case():
    """Check unsupported native case names raise a clear error."""

    with pytest.raises(ValueError, match="Unknown native FAST case"):
        native_case("Demo")
